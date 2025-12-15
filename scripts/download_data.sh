#!/bin/bash

# Download molecular property prediction datasets
# MoleculeNet benchmark datasets

set -e

echo "=============================================="
echo "Chandankumar is Downloading Molecular Datasets"
echo "=============================================="

# Create directories
mkdir -p ../data/raw/{BBBP,HIV,QM9,ESOL,FreeSolv}

# Function to download and extract
download_dataset() {
    local name=$1
    local url=$2
    local output_dir=$3
    
    echo ""
    echo "Downloading $name dataset..."
    
    if [ -f "$output_dir/${name}.csv" ]; then
        echo "$name already exists. Skipping..."
        return
    fi
    
    wget -q --show-progress -O "$output_dir/${name}.csv" "$url"
    echo " $name downloaded successfully"
}

# BBBP (Blood-Brain Barrier Penetration) - Classification
echo ""
echo "1. BBBP Dataset (Blood-Brain Barrier Penetration)"
echo "   - Task: Binary Classification"
echo "   - Size: ~2000 molecules"
echo "   - Target: Penetrates BBB (0/1)"
download_dataset "BBBP" \
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv" \
    "../data/raw/BBBP"

# HIV (HIV Replication Inhibition) - Classification  
echo ""
echo "2. HIV Dataset (HIV Replication Inhibition)"
echo "   - Task: Binary Classification"
echo "   - Size: ~41000 molecules"
echo "   - Target: Inhibits HIV replication (0/1)"
download_dataset "HIV" \
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv" \
    "../data/raw/HIV"

# ESOL (Water Solubility) - Regression
echo ""
echo "3. ESOL Dataset (Water Solubility)"
echo "   - Task: Regression"
echo "   - Size: ~1100 molecules"
echo "   - Target: Log solubility in mols/L"
download_dataset "ESOL" \
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv" \
    "../data/raw/ESOL"

# FreeSolv (Hydration Free Energy) - Regression
echo ""
echo "4. FreeSolv Dataset (Hydration Free Energy)"
echo "   - Task: Regression"
echo "   - Size: ~640 molecules"
echo "   - Target: Hydration free energy (kcal/mol)"
download_dataset "FreeSolv" \
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv" \
    "../data/raw/FreeSolv"

# QM9 (Quantum Mechanics Properties) - Multi-task Regression
echo ""
echo "5. QM9 Dataset (Quantum Mechanics)"
echo "   - Task: Multi-task Regression"
echo "   - Size: ~130000 molecules"
echo "   - Multiple quantum properties"
echo "   Note: This is a large dataset, may take time..."

QM9_DIR="../data/raw/QM9"
if [ -f "$QM9_DIR/qm9.csv" ]; then
    echo "QM9 already exists. Skipping..."
else
    # QM9 is typically downloaded via PyTorch Geometric
    echo "QM9 will be downloaded automatically when first accessed via PyTorch Geometric"
    echo "Alternatively, download from: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
fi

echo ""
echo "Download Complete!"