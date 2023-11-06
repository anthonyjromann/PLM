import pandas as pd
from Bio import SeqIO
import os
from esm import pretrained
import torch

# Input directory
fastaDir = 'C:/Users/tuk48434/Desktop/pf00001'


def run_ESM_automated(sequence):
    """Run a protein sequence using ESM-1b and return a dataframe containing associated probability values.

    Keyword arguments:
    sequence -- the protein sequence, represented by a string
    """
    # Remove all gaps '-' from the sequence.
    sequence = sequence.replace('-', '')

    # Package protein sequence in batch converter format (tuple nested in an array), then convert to a vector using the batch converter.
    data = [('protein', sequence)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Perform inference and softmax function to  correctly prepare data.
    with torch.no_grad():
        logits = model(batch_tokens)["logits"]
        aa_logits = logits[0, :, aa_indices]  # select amino acid logits only
        token_probs = torch.softmax(aa_logits, dim=-1)

    # Remove the first and last token (they empty placeholders for padding).
    token_probs = token_probs[1:-1]
    new_token_probs = token_probs.cpu().numpy()

    # Convert to dataframe and return.
    df = pd.DataFrame(new_token_probs, columns=list('ACDEFGHIKLMNPQRSTVWY'))
    return df

# Initialize model for inferenc
model, alphabet = pretrained.esm1b_t33_650M_UR50S()
aa_indices = [alphabet.get_idx(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'] 
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# For all MSA files in the selected directory, extract the sequence and species name before performing their inference and saving their output.
for filename in os.listdir(fastaDir):
    with open(os.path.join(fastaDir, filename), "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            species = str(record.description)
            output_df = run_ESM_automated(sequence)
            name1 = filename.split('_')
            name = name1[1] + '_' + name1[2].split('-')[0]
            output_name = 'C:/Users/tuk48434/Desktop/pf1_esm1/' + name + '_' + species + '.csv'
            output_df.to_csv(output_name, index=False)

        
