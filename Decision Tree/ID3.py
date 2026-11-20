import math
from collections import Counter
# Bộ dữ liệu Play Tennis (14 mẫu): [Thoi_Tiet, Nhiet_Do, Do_Am, Gio, Choi_Tennis]
dataset = [
    ["Nang", "Nong", "Cao", "Yeu", "No"],
    ["Nang", "Nong", "Cao", "Manh", "No"],
    ["U_Am", "Nong", "Cao", "Yeu", "Yes"],
    ["Mua",  "Mat",  "Cao", "Yeu", "Yes"],
    ["Mua",  "Lanh", "Binh_Thuong", "Yeu", "Yes"],
    ["Mua",  "Lanh", "Binh_Thuong", "Manh", "No"],
    ["U_Am", "Lanh", "Binh_Thuong", "Manh", "Yes"],
    ["Nang", "Mat",  "Cao", "Yeu", "No"],
    ["Nang", "Lanh", "Binh_Thuong", "Yeu", "Yes"],
    ["Mua",  "Mat",  "Binh_Thuong", "Yeu", "Yes"],
    ["Nang", "Mat",  "Binh_Thuong", "Manh", "Yes"],
    ["U_Am", "Mat",  "Cao", "Manh", "Yes"],
    ["U_Am", "Nong", "Binh_Thuong", "Yeu", "Yes"],
    ["Mua",  "Mat",  "Cao", "Manh", "No"],
]

feature_names = ["Thoi_Tiet", "Nhiet_Do", "Do_Am", "Gio"]

def calc_entropy(y):
    size = len(y)
    if size == 0: return 0.0
    entropy = 0
    for i in Counter(y).values():
        p = i / size
        if p == 0: continue
        entropy -= p*math.log2(p)
    return entropy

def calc_info_gain(data, feature_idx):
    labels = [row[-1] for row in data]
    base_Entropy = calc_entropy(labels)
    subsets = {}
    for row in data:   
        if row[feature_idx] not in subsets: subsets[row[feature_idx]] = []
        label = row[-1]
        subsets[row[feature_idx]].append(label)
    weighted_entropy = 0.0
    for sub_label in subsets.values():
        weight = len(sub_label) / len(data)
        weighted_entropy += weight*calc_entropy(sub_label)
    return base_Entropy - weighted_entropy

def build_id3(data, features):
    label = [row[-1] for row in data]
    
    if len(set(label)) == 1: return label[0]
    if len(features) == 0: return Counter(label).most_common(1)[0][0]
    
    gains = [calc_info_gain(data, idx) for idx in range(len(features))]
    best_idx = gains.index(max(gains))
    best_gain = gains[best_idx]

    if best_gain == 0: return Counter(label).most_common(1)[0][0]

    sub_feature = features[:best_idx] + features[best_idx+1:]
    best_feature = features[best_idx]
    tree = {best_feature: {}}
    unique = list(dict.fromkeys(row[best_idx] for row in data))

    for val in unique:
        sub_data = []
        for row in data:
            if row[best_idx] == val:
                new_row = row[:best_idx] + row[best_idx+1:]
                sub_data.append(new_row)
        tree[best_feature][val] = build_id3(sub_data, sub_feature)
    return tree
def predict_one(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root_feature = next(iter(tree))
    feature_val = sample.get(root_feature)

    if feature_val not in tree[root_feature]:
        return None
    return predict_one(tree[root_feature][feature_val], sample)

if __name__ == "__main__":
    tree = build_id3(dataset, feature_names)
    test_sample = {
        "Thoi_Tiet": "Nang",
        "Nhiet_Do": "Mat",
        "Do_Am": "Cao",
        "Gio": "Yeu",
    }
    print("\nMẫu kiểm thử:", test_sample)
    print("Dự đoán kết quả:", predict_one(tree, test_sample))