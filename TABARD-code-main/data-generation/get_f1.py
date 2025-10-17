import json
import sys
import os

# **Provide the folder paths directly in the code**
predictions_folder_path = r"C:\Users\MAMANROY CHOUDHURY\Downloads\New folder (6)"
labels_folder_path = r"C:\Users\MAMANROY CHOUDHURY\Downloads\New folder (5)"

def get_score(predictions_folder_path, labels_folder_path):
    prediction_files = sorted(os.listdir(predictions_folder_path))
    label_files = sorted(os.listdir(labels_folder_path))
    prediction_files = [file for file in prediction_files if file.endswith('.json') and os.path.isfile(os.path.join(predictions_folder_path, file))]
    label_files = [file for file in label_files if file.endswith('.json') and os.path.isfile(os.path.join(labels_folder_path, file))]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for prediction_file in prediction_files:
        prediction_file_path = os.path.join(predictions_folder_path, prediction_file)
        label_file_path = os.path.join(labels_folder_path, prediction_file)
        with open(prediction_file_path, 'r') as prediction_file, open(label_file_path, 'r') as label_file:
            try:
                predictions = json.load(prediction_file)
                labels = json.load(label_file)
                
                for prediction, label in zip(predictions, labels):
                    for prediction_key, label_key in zip(prediction, label):
                        # print(prediction[prediction_key], label[label_key])
                        if prediction[prediction_key] == "Yes" and label[label_key] == "Yes":
                            true_positives += 1
                        elif prediction[prediction_key] == "Yes" and label[label_key] == "No":
                            false_positives += 1
                        elif prediction[prediction_key] == "No" and label[label_key] == "Yes":
                            false_negatives += 1
            except json.JSONDecodeError as e:
                print(f"Error reading {prediction_file}: {e}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(true_positives, false_positives, false_negatives)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# **No longer using command-line arguments**
if __name__ == "__main__":
    get_score(predictions_folder_path, labels_folder_path)
