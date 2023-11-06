from esm import pretrained
import torch
import pandas as pd
from Bio import SeqIO
import os


def fasta_to_tuples(filename, position_to_mask=None):
    """ Convert the input data into a usable format for the batch converter.

    Keyword arguments:
    filename -- the path to the data file
    position_to_mask -- if masking is performed, this is the zero-based index of where the sequence will be masked (default there is no masking)
    """
    list_of_tuples = []

    with open(filename, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            
            # Check if the description matches 'Homo_sapiens' and a mask position is provided
            if "Homo_sapiens" in record.description and position_to_mask is not None:
                # Modify the sequence to replace the amino acid at the given position with '<mask>'
                new_seq = sequence[:position_to_mask] + '<mask>' + sequence[position_to_mask + 1:]
            # When masking occurs, ensure that the extra characters added from the token are padded with gaps
            elif position_to_mask is not None:
                new_seq = sequence + '-----'
            else:
                new_seq = sequence
            list_of_tuples.append((record.id, new_seq))
    return list_of_tuples

# Initialize model for inference
model, alphabet = pretrained.esm_msa1b_t12_100M_UR50S()
aa_indices = [alphabet.get_idx(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'] 
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# Input and output directories
directory = 'C:/Users/tuk48434/Desktop/pf00001'
output = 'C:/Users/tuk48434/Desktop/PF1_MSA_new'

for filename in os.listdir(directory):
    # Use zero-based index
    protein_name = filename.split('_')[2]
    value = protein_name.split('-')[0]
    
    # Convert data to vectors for model inference
    data = fasta_to_tuples(os.path.join(directory, filename), position_to_mask=None)
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Perform inference and output softmax probabilities for the hg19 sequence (the first sequence in the alignment file)
    with torch.no_grad():
        logits = model(batch_tokens)["logits"]
        aa_logits = logits[0, :, :, aa_indices]  # select amino acid logits only
        token_probs = torch.softmax(aa_logits, dim=-1)

    # Save output file
    new_token_probs = token_probs.cpu().numpy()
    seq = new_token_probs[0]
    df = pd.DataFrame(seq[1:], columns=list('ACDEFGHIKLMNPQRSTVWY'))
    output_name = 'C:/Users/tuk48434/Desktop/PF1_MSA_new/' + 'hg19_NM_' + value + '.csv'
    df.to_csv(output_name, index=False)
