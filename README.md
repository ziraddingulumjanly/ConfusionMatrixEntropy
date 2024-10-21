The entropy between two classes (i and j) can be calculated using the following formula:

e_ij = - [ (C_ij / (C_ij + C_ji)) * ln(C_ij / (C_ij + C_ji)) + (C_ji / (C_ij + C_ji)) * ln(C_ji / (C_ij + C_ji)) ]

Where:
- C_ij is the number of times class i was predicted as class j.
- C_ji is the number of times class j was predicted as class i.
