import os
import re
import csv

root_dir = 'attack_results'
output_file = 'aggregated_report.csv'

save_path = os.path.join("./", "reports")
os.makedirs(save_path, exist_ok=True)

output_file = os.path.join(save_path, output_file)

rows = []
header = ['Train Type', 'PIM Type', 'Attack', 'Epsilon', 'Clean Accuracy (%)', 'Robust Accuracy (%)', 'Attack Success Rate (%)']

pattern_accuracy = {
    'clean': re.compile(r'Clean Accuracy:\s+([\d.]+)%'),
    'robust': re.compile(r'Robust Accuracy.*?:\s+([\d.]+)%'),
    'success': re.compile(r'Attack Success Rate:\s+([\d.]+)%'),
}

for train_type in ['adversarial_train', 'normal_train']:
    for pim_type in ['pim', 'non_pim']:
        for attack in ['fgsm', 'ifgsm', 'deepfool', 'pgd']:
            attack_path = os.path.join(root_dir, train_type, pim_type, attack)
            if not os.path.exists(attack_path):
                continue

            for filename in os.listdir(attack_path):
                match = re.match(r'analysis_epsilon([\d.]+)\.txt', filename)
                if not match:
                    continue
                epsilon = float(match.group(1))
                file_path = os.path.join(attack_path, filename)

                with open(file_path, 'r') as f:
                    text = f.read()

                    try:
                        clean_acc = float(pattern_accuracy['clean'].search(text).group(1))
                        robust_acc = float(pattern_accuracy['robust'].search(text).group(1))
                        attack_success = float(pattern_accuracy['success'].search(text).group(1))
                    except AttributeError:
                        print(f"Skipping malformed file: {file_path}")
                        continue

                    rows.append([
                        train_type,
                        pim_type,
                        attack,
                        epsilon,
                        clean_acc,
                        robust_acc,
                        attack_success
                    ])

# Write to CSV
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(sorted(rows, key=lambda x: (x[0], x[1], x[2], x[3])))

print(f"Report saved to {output_file}")
