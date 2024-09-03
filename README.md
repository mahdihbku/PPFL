# A Lightweight Committee-Based Approach for Privacy-Preserving Federated Learning

This is a proof-of-concept for our work submitted to CCNC2025.


# Requirements

Download dataset from https://www.kaggle.com/datasets/stav42/dataset-bosch
Split database into chunks:
```bash
python split.py
```

# Execution commands

Start server :
```bash
python PPserver.py
```

Start clients :
```bash
python PPclient MY_INDEX
```

Start users :
```bash
python PPcommitteeMember.py MY_LISTENING_PORT
```
