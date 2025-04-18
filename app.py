from flask import Flask, request, render_template
import numpy as np
import joblib

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型
model = joblib.load("model.pkl")

# 定义氨基酸特征映射
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
aa_properties = {
    'A': {'hydrophobicity': 1.8, 'polarity': 9.87, 'size': 89.1},
    'V': {'hydrophobicity': 4.2, 'polarity': 9.62, 'size': 117.1},
    'L': {'hydrophobicity': 3.8, 'polarity': 9.60, 'size': 131.2},
    'I': {'hydrophobicity': 4.5, 'polarity': 9.60, 'size': 131.2},
    'F': {'hydrophobicity': 2.8, 'polarity': 9.24, 'size': 165.2},
    'W': {'hydrophobicity': -0.9, 'polarity': 9.41, 'size': 204.2},
    'P': {'hydrophobicity': -1.6, 'polarity': 10.64, 'size': 115.1},
    'M': {'hydrophobicity': 1.9, 'polarity': 9.21, 'size': 149.2},
    'D': {'hydrophobicity': -3.5, 'polarity': 3.90, 'size': 133.1},
    'E': {'hydrophobicity': -3.5, 'polarity': 4.25, 'size': 147.1},
    'S': {'hydrophobicity': -0.8, 'polarity': 13.00, 'size': 105.1},
    'T': {'hydrophobicity': -0.7, 'polarity': 13.00, 'size': 119.1},
    'Y': {'hydrophobicity': -1.3, 'polarity': 9.11, 'size': 181.2},
    'K': {'hydrophobicity': -3.9, 'polarity': 10.53, 'size': 146.2},
    'R': {'hydrophobicity': -4.5, 'polarity': 12.48, 'size': 174.2},
    'H': {'hydrophobicity': -3.2, 'polarity': 6.04, 'size': 155.2},
}

def seq_to_features_with_properties(sequence, max_seq_length=200):
    features_size = (len(amino_acids) ** 2) + (len(amino_acids) * 3) + 400
    features = np.zeros(features_size)

    for i in range(len(sequence) - 1):
        aa1 = sequence[i]
        aa2 = sequence[i + 1]
        if aa1 in aa_to_int and aa2 in aa_to_int:
            index = aa_to_int[aa1] * len(amino_acids) + aa_to_int[aa2]
            features[index] += 1

    for i, aa in enumerate(sequence):
        if i >= max_seq_length:
            break
        if aa in aa_properties:
            properties = aa_properties[aa]
            features[len(amino_acids) ** 2 + i * 3] = properties['hydrophobicity']
            features[len(amino_acids) ** 2 + i * 3 + 1] = properties['polarity']
            features[len(amino_acids) ** 2 + i * 3 + 2] = properties['size']
        features[len(amino_acids) ** 2 + len(amino_acids) * 3 + i] = i / len(sequence)
    return features

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        seq = request.form["sequence"]
        features = seq_to_features_with_properties(seq)
        pred = model.predict([features])[0]
        result = "Umami" if pred == 1 else "Not Umami"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
