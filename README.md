# MS-CPI:Multi-scale learning with the integration of pharmacophore information for predicting compound-protein interaction


## MS-CPI



<img width="1304" alt="multiscale-model" src="https://github.com/liyf233/MS-CPI/assets/47840818/a8a7a974-0483-4f3f-99c2-425981d4523e">


## Setup and dependencies 

Dependencies:
- python 3.6
- pytorch >= 1.2.0
- numpy
- RDkit = 2019.03.3.0
- pandas
- Gensim >=3.4.0

### Datasets

The datasets utilized in our research are stored within the "data" directory.

### Instructions

1. **Preprocess Raw Data:**
   Execute the following command to preprocess the raw data, extracting atom features, and converting amino acid sequences to embeddings:

   ```bash
   python mol_featurizer.py
   ```

   Ensure to review the script (`mol_featurizer.py`) for any additional configuration options that may need customization according to specific requirements.

2. **Train and Evaluate MS-CPI:**
   To train and assess MS-CPI on benchmark datasets, use the following command:

   ```bash
   python main.py
   ```

   Similarly, check the script (`main.py`) for potential adjustable parameters or configurations based on your needs.
