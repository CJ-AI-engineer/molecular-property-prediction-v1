"""
Flask Web Application for Molecular Property Prediction
Predicts blood-brain barrier penetration from SMILES strings
"""

from flask import Flask, render_template, request, jsonify
import torch
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import MolecularGCN, MolecularGIN, MolecularGAT
from src.data import MoleculeGraphBuilder, AtomFeaturizer, BondFeaturizer
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models
MODELS = {}
GRAPH_BUILDER = None
DEVICE = 'cpu'

def load_models():
    """Load all available trained models."""
    global MODELS, GRAPH_BUILDER, DEVICE
    
    # Initialize graph builder
    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondFeaturizer()
    GRAPH_BUILDER = MoleculeGraphBuilder(atom_featurizer, bond_featurizer)
    
    # Check for GPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Try to load models from checkpoints
    checkpoint_dir = project_root / 'checkpoints'
    
    if checkpoint_dir.exists():
        # Look for model checkpoints
        model_configs = {
            'GCN': {
                'class': MolecularGCN,
                'pattern': '*gcn*/best_model.pt',
                'name': 'Graph Convolutional Network'
            },
            'GIN': {
                'class': MolecularGIN,
                'pattern': '*gin*/best_model.pt',
                'name': 'Graph Isomorphism Network'
            },
            'GAT': {
                'class': MolecularGAT,
                'pattern': '*gat*/best_model.pt',
                'name': 'Graph Attention Network'
            }
        }
        
        for model_key, config in model_configs.items():
            checkpoints = list(checkpoint_dir.glob(config['pattern']))
            if checkpoints:
                try:
                    checkpoint_path = checkpoints[0]  # Use first match
                    print(f"Loading {model_key} from {checkpoint_path}")
                    
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                    
                    # Get model configuration from checkpoint
                    model_state = checkpoint['model_state_dict']
                    
                    # Infer dimensions from checkpoint state dict
                    # Get first conv layer weights to determine input dimensions
                    first_layer_key = None
                    edge_layer_key = None
                    
                    for key in model_state.keys():
                        if 'conv' in key.lower() and 'weight' in key:
                            if first_layer_key is None:
                                first_layer_key = key
                            if 'edge' in key.lower() and edge_layer_key is None:
                                edge_layer_key = key
                    
                    if first_layer_key:
                        first_layer_shape = model_state[first_layer_key].shape
                        
                        # For GCN: convs.0.lin.weight has shape [hidden_dim, node_feat_dim]
                        # For GIN: convs.0.nn.0.weight has shape [hidden_dim, node_feat_dim]
                        # For GAT: similar pattern
                        
                        if model_key == 'GCN':
                            hidden_dim = first_layer_shape[0]
                            node_feat_dim = first_layer_shape[1]
                        elif model_key == 'GIN':
                            hidden_dim = first_layer_shape[0]
                            node_feat_dim = first_layer_shape[1]
                        else:  # GAT
                            # GAT might have different structure
                            hidden_dim = checkpoint.get('hidden_dim', 128)
                            node_feat_dim = first_layer_shape[-1]  # Last dimension
                        
                        # Try to get edge feature dim from edge layers
                        if edge_layer_key:
                            edge_layer_shape = model_state[edge_layer_key].shape
                            edge_feat_dim = edge_layer_shape[-1]  # Last dimension is edge features
                        else:
                            edge_feat_dim = checkpoint.get('edge_feat_dim', 10)
                    else:
                        # Fallback to checkpoint metadata if available
                        hidden_dim = checkpoint.get('hidden_dim', 128)
                        node_feat_dim = checkpoint.get('node_feat_dim', 50)
                        edge_feat_dim = checkpoint.get('edge_feat_dim', 10)
                    
                    num_layers = checkpoint.get('num_layers', 5)
                    
                    print(f"  Model config: node_feat={node_feat_dim}, edge_feat={edge_feat_dim}, hidden={hidden_dim}, layers={num_layers}")
                    
                    # Create model with correct dimensions
                    model = config['class'](
                        node_feat_dim=node_feat_dim,
                        edge_feat_dim=edge_feat_dim,
                        hidden_dim=hidden_dim,
                        num_tasks=1,
                        num_layers=num_layers
                    )
                    
                    # Load weights
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(DEVICE)
                    model.eval()
                    
                    MODELS[model_key] = {
                        'model': model,
                        'name': config['name'],
                        'path': str(checkpoint_path)
                    }
                    print(f" {model_key} loaded successfully")
                    
                except Exception as e:
                    print(f"✗ Failed to load {model_key}: {e}")
    
    if not MODELS:
        print("⚠ Warning: No models loaded. Please train models first.")
    else:
        print(f"\n Loaded {len(MODELS)} model(s): {', '.join(MODELS.keys())}")

def smiles_to_graph(smiles):
    """Convert SMILES to graph data."""
    try:
        # Convert SMILES to graph directly
        data = GRAPH_BUILDER.smiles_to_graph(smiles)
        if data is None:
            return None, "Failed to convert molecule to graph"
        
        return data, None
    except Exception as e:
        return None, str(e)

def predict_single(smiles, model_name='GCN'):
    """Make prediction for a single molecule."""
    if model_name not in MODELS:
        return {'error': f'Model {model_name} not available'}
    
    # Convert SMILES to graph
    data, error = smiles_to_graph(smiles)
    if error:
        return {'error': error}
    
    # Make prediction
    try:
        data = data.to(DEVICE)
        model = MODELS[model_name]['model']
        
        with torch.no_grad():
            prediction = model(data)
            probability = torch.sigmoid(prediction).item()
        
        # Get molecule properties
        mol = Chem.MolFromSmiles(smiles)
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        return {
            'probability': probability,
            'prediction': 'Penetrates BBB' if probability >= 0.5 else 'Does Not Penetrate BBB',
            'confidence': abs(probability - 0.5) * 2,  # 0 to 1 scale
            'molecular_weight': round(mol_weight, 2),
            'logp': round(logp, 2),
            'model': model_name
        }
    
    except Exception as e:
        return {'error': str(e)}

def draw_molecule(smiles, size=(300, 300)):
    """Draw molecule and return as base64 image."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=size)
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Error drawing molecule: {e}")
        return None

@app.route('/')
def index():
    """Main page."""
    available_models = list(MODELS.keys())
    return render_template('index.html', models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint."""
    data = request.get_json()
    smiles = data.get('smiles', '').strip()
    model_name = data.get('model', 'GCN')
    
    if not smiles:
        return jsonify({'error': 'No SMILES provided'}), 400
    
    # Make prediction
    result = predict_single(smiles, model_name)
    
    if 'error' in result:
        return jsonify(result), 400
    
    # Add molecule image
    result['molecule_image'] = draw_molecule(smiles)
    result['smiles'] = smiles
    
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    data = request.get_json()
    smiles_list = data.get('smiles_list', [])
    model_name = data.get('model', 'GCN')
    
    if not smiles_list:
        return jsonify({'error': 'No SMILES provided'}), 400
    
    results = []
    for smiles in smiles_list:
        result = predict_single(smiles.strip(), model_name)
        result['smiles'] = smiles
        results.append(result)
    
    return jsonify({'results': results})

@app.route('/models')
def models():
    """Get available models."""
    models_info = {}
    for key, value in MODELS.items():
        models_info[key] = {
            'name': value['name'],
            'path': value['path']
        }
    return jsonify(models_info)

@app.route('/examples')
def examples():
    """Get example molecules."""
    examples = [
        {
            'name': 'Caffeine',
            'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'description': 'Common stimulant, crosses BBB'
        },
        {
            'name': 'Aspirin',
            'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'description': 'Pain reliever, limited BBB penetration'
        },
        {
            'name': 'Dopamine',
            'smiles': 'C1=CC(=C(C=C1CCN)O)O',
            'description': 'Neurotransmitter, does not cross BBB'
        },
        {
            'name': 'L-DOPA',
            'smiles': 'C1=CC(=C(C=C1CC(C(=O)O)N)O)O',
            'description': 'Dopamine precursor, crosses BBB'
        },
        {
            'name': 'Morphine',
            'smiles': 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O',
            'description': 'Opioid analgesic, crosses BBB'
        }
    ]
    return jsonify(examples)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS),
        'available_models': list(MODELS.keys()),
        'device': DEVICE
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Molecular Property Prediction Web App")
    print("=" * 60)
    print("\nLoading models...")
    load_models()
    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print("Access the app at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

