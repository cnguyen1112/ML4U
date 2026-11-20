import math


def cross_Loss_Entropy(data, guess):
    eps = 1e-15
    if not data: return 0.0
    entropy = 0.0
    for i in range(len(data)):
        temp = 0
        for j in range(len(data[0])):
            p = max(eps, min(1.0 - eps, guess[i][j]))
            temp += data[i][j]*math.log(p)
        entropy += temp
    loss_entropy = entropy /(-len(data))
    return loss_entropy

if __name__ == "__main__":
    y_true = [
        [1, 0, 0],  # Mẫu 1: lớp 0
        [0, 1, 0],  # Mẫu 2: lớp 1
        [0, 0, 1],  # Mẫu 3: lớp 2
    ]
    y_pred = [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
    ]

    loss = cross_Loss_Entropy(y_true, y_pred)
    # Kỳ vọng: (-ln(0.7) - ln(0.8) - ln(0.5)) / 3 ≈ 0.424322
    assert math.isclose(loss, 0.424322, rel_tol=1e-4), f"Sai giá trị loss: {loss}"
    print(f"PASS Bài 2! Batch Cross-Entropy Loss = {loss:.6f}")