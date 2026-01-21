import os
import tempfile
import subprocess



def paired_positions(dot_bracket):
    stack = []
    pairs = []
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                pairs.append((stack.pop(), i))
    return pairs


def predict_ss(sequence, output_dir):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.fa', delete=False) as tmp_fasta:
            tmp_fasta.write(f'>seq temp\n{sequence}\n')
            tmp_fasta.flush()  # Make sure all data is written
            cmd = [
                '/home/nagle/final_version/EternaFold/src/contrafold', 'predict', tmp_fasta.name
                ]
        
            # Execute the command and capture stdout and stderr

                
            process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            # Process the captured stdout as the prediction result
            if process.returncode == 0:
            
                prediction = process.stdout.strip()
            
                #ONLY THE DOT BRACKET FOR NOW
                prediction = prediction.split('\n')[-1]
                #REMOVE the score as well
                score = prediction.rsplit(' ')[-1]
                prediction = prediction.rsplit(' ')[0]
            
                output_file_path = os.path.join(output_dir, f"{sequence}.txt")
                f = open(output_file_path, "w")
                f.write(prediction)
                f.close()
            else:
                print(f"Error in prediction for {sequence}: {process.stderr}")