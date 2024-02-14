# LambdaLabs Labs (16 GB GPU to 48 GB GPU) (Training)

| Instance            | GPU                | GPU Memory | vCPUs | RAM   | Storage | Price Per Hour |
|---------------------|--------------------|------------|-------|-------|---------|----------------|
| 1x NVIDIA A10       | 1                  | 24 GB      | 30    | 200 GB | 1.4 TB  | $0.60 / hr     |
| 1x Quadro RTX 6000  | 1                  | 24 GB      | 14    | 46 GB  | 512 GB | $0.50 / hr     |
| 8x Tesla V100       | 1                  | 16 GB      | 92    | 448 GB | 5.9 TB  | $4.40 / hr     |
| 1x NVIDIA A100      | 1                  | 40 GB      | 30    | 200 GB | 512 GB | $1.10 / hr     |
| 2x NVIDIA A100      | 1                  | 40 GB      | 60    | 400 GB | 1 TB   | $2.20 / hr     |
| 4x NVIDIA A100      | 1                  | 40 GB      | 120   | 800 GB | 1 TB   | $4.40 / hr     |
| 8x NVIDIA A100      | 1                  | 40 GB      | 124   | 1800 GB| 6 TB   | $8.80 / hr     |
| 1x NVIDIA RTX A6000 | 1                  | 48 GB      | 14    | 100 GB | 200 GB | $0.80 / hr     |
| 2x NVIDIA RTX A6000 | 1                  | 48 GB      | 28    | 200 GB | 1 TB   | $1.60 / hr     |
| 4x NVIDIA RTX A6000 | 1                  | 48 GB      | 56    | 400 GB | 1 TB   | $3.20 / hr     |

# Lambda Labs (80 GB GPU) (Training)
| Instance            | GPU                | GPU Memory | vCPUs | RAM   | Storage | Price Per Hour |
|---------------------|--------------------|------------|-------|-------|---------|----------------|
| 1x NVIDIA H100 PCIe | 1                  | 80 GB      | 26    | 200 GB | 512 TB | $2.40 / hr     |
| 8x NVIDIA A100      | 1                  | 80 GB      | 240   | 1800 GB| 20 GB  | $12.00 / hr    |

# EC2 p4 Instance Type (Training)
| Instance Size    | GPU  | GPU Memory | vCPUs | RAM     | Storage         | On-Demand Hourly Cost |
|------------------|------|------------|-------|---------|-----------------|-----------------------|
| p4d.24xlarge     | 8    | 320 GB HBM2| 96    | 1152 GB | 8 x 1000 NVMe SSD | $36.04986 / hr        |

# EC2 p3 Instance Type (Training)
| Instance Size    | GPU  | GPU Memory | vCPUs | RAM     | Storage         | On-Demand Hourly Cost |
|------------------|------|------------|-------|---------|-----------------|-----------------------|
| p3.2xlarge      | 1   | 16 GB      | 8     | 61 GB   | EBS only | $3.259 / hr     |
| p3.8xlarge      | 4   | 64 GB      | 32    | 244 GB  | EBS only | $13.036 / hr    |
| p3.16xlarge     | 8   | 128 GB     | 64    | 488 GB  | EBS only | $24.48 / hr     |

# EC2 g5 Instance Type (Training)

| Instance Type | GPU Count | GPU Memory | vCPUs | RAM (GB) | Storage          | On-Demand Hourly Cost |
|---------------|-----------|------------|-------|----------|------------------|-----------------------|
| g5.xlarge     | 1         | 24         | 4     | 16       | 1 x 250 NVMe SSD | $1.1066/ hr           |
| g5.2xlarge    | 1         | 24         | 8     | 32       | 1 x 450 NVMe SSD | $1.3332/ hr           |
| g5.4xlarge    | 1         | 24         | 16    | 64       | 1 x 600 NVMe SSD | $1.7864/ hr           |
| g5.8xlarge    | 1         | 24         | 32    | 128      | 1 x 900 NVMe SSD | $2.6928/ hr           |
| g5.16xlarge   | 1         | 24         | 64    | 256      | 1 x 1900 NVMe SSD| $4.5056/ hr           |
| g5.12xlarge   | 4         | 96         | 48    | 192      | 1 x 3800 NVMe SSD| $6.2392/ hr           |
| g5.24xlarge   | 4         | 96         | 96    | 384      | 1 x 3800 NVMe SSD| $8.9584/ hr           |
| g5.48xlarge   | 4         | 192        | 192   | 768      | 2x 3800 NVMe SSD | $17.9168/ hr          |

# EC2 G4dn Instance Type (Training)
| Instance Size | GPU | GPU Memory | vCPUs | RAM     | Storage             | On-Demand Hourly Cost |
|---------------|-----|------------|-------|---------|---------------------|-----------------------|
| g4dn.xlarge   | 1   | 16 GB      | 4     | 16 GB   | 1 x 125 NVMe SSD   | $0.558/ hr            |
| g4dn.2xlarge  | 1   | 16 GB      | 8     | 32 GB   | 1 x 225 NVMe SSD   | $0.797/ hr            |
| g4dn.4xlarge  | 1   | 16 GB      | 16    | 64 GB   | 1 x 225 NVMe SSD   | $1.276/ hr            |
| g4dn.8xlarge  | 1   | 16 GB      | 32    | 128 GB  | 1 x 900 NVMe SSD   | $2.307/ hr            |
| g4dn.16xlarge | 1   | 16 GB      | 64    | 256 GB  | 1 x 900 NVMe SSD   | $4.147/ hr            |
| g4dn.12xlarge | 4   | 64 GB      | 48    | 192 GB  | 1 x 900 NVMe SSD   | $4.613/ hr            |
| g4dn.metal    | 8   | 64 GB      | 96    | 384 GB  | 2 x 900 NVMe SSD   | $7.824/ hr            |

# EC2 G4ad Instance Type (Training)
| Instance Size | GPU | GPU Memory | vCPUs | RAM     | Storage              | On-Demand Hourly Cost |
|---------------|-----|------------|-------|---------|----------------------|-----------------------|
| g4ad.xlarge   | 1   | 8 GB       | 4     | 16 GB   | 1 x 150 NVMe SSD    | $0.41638/ hr          |
| g4ad.2xlarge  | 1   | 8 GB       | 8     | 32 GB   | 1 x 300 NVMe SSD    | $0.59529/ hr          |
| g4ad.4xlarge  | 1   | 8 GB       | 16    | 64 GB   | 1 x 600 NVMe SSD    | $0.9537/ hr           |
| g4ad.8xlarge  | 2   | 16 GB      | 32    | 128 GB  | 1 x 1200 NVMe SSD   | $1.9074/ hr           |
| g4ad.16xlarge | 4   | 32 GB      | 64    | 256 GB  | 1 x 2400 NVMe SSD   | $3.8148/ hr           |

# EC2 Trn1 Instance Type (Training)
| Instance Size | Trainium Accelerators | Accelerator Memory | vCPUs | RAM     | Storage              | On-Demand Hourly Cost |
|---------------|-----------------------|--------------------|-------|---------|----------------------|-----------------------|
| trn1.2xl      | 1                     | 32 GB              | 8     | 32 GB   | 1 x 500 NVMe SSD    | $1.47813/ hr          |
| trn1.32xl     | 16                    | 512 GB             | 128   | 512 GB  | 4 x 2000 NVMe SSD   | $23.65/ hr            |

# EC2 g5g Instance Type (Inference)
| Instance Size | NVIDIA T4G Tensor Core GPU | GPU Memory | vCPUs | RAM     | On-Demand Hourly Cost |
|---------------|---------------------------|------------|-------|---------|-----------------------|
| g5g.xlarge    | 1                         | 16 GB      | 4     | 8 GB   | $0.4452/ hr           |
| g5g.2xlarge   | 1                         | 16 GB      | 8     | 16 GB  | $0.5894/ hr           |
| g5g.4xlarge   | 1                         | 16 GB      | 16    | 32 GB  | $0.8777/ hr           |
| g5g.8xlarge   | 1                         | 16 GB      | 32    | 64 GB  | $1.4543/ hr           |
| g5g.16xlarge  | 2                         | 32 GB      | 64    | 128 GB | $2.744/ hr            |
| g5g.metal     | 2                         | 32 GB      | 64    | 128 GB | $2.744/ hr            |

# EC2 Inf1 Instance Type (Inference) 

| Instance Size | Inferentia Accelerators | Inter-accelerator Interconnect | vCPUs | RAM | Storage | On-Demand Hourly Cost |
|---------------|-------------------------|-------------------------------|-------|-----|---------|-----------------------|
| inf1.xlarge   | 1                       | N/A                           | 4     | 8   | EBS only| $0.242/ hr            |
| inf1.2xlarge  | 1                       | N/A                           | 8     | 16  | EBS only| $0.384/ hr            |
| inf1.6xlarge  | 4                       | N/A                           | 24    | 48  | EBS only| $1.251/ hr            |
| inf1.24xlarge | 16                      | N/A                           | 96    | 192 | EBS only| $5.005/ hr            |

# EC2 Inf2 Instance Type (Inference)
| Instance Size | Inferentia Accelerators | Accelerator Memory | Inter-accelerator Interconnect | vCPUs | RAM     | Storage | On-Demand Hourly Cost |
|---------------|-------------------------|--------------------|-------------------------------|-------|---------|---------|-----------------------|
| inf2.xlarge   | 1                       | 32 GB              | N/A                           | 4     | 16 GB   | EBS only| $0.83402/ hr          |
| inf2.8xlarge  | 1                       | 32 GB              | N/A                           | 32    | 128 GB  | EBS only| $2.16465/ hr          |
| inf2.24xlarge | 6                       | 192 GB             | N/A                           | 96    | 384 GB  | EBS only| $7.1397/ hr           |
| inf2.48xlarge | 12                      | 384 GB             | N/A                           | 192   | 468 GB  | EBS only| $14.27939/ hr         |

# Azure 
| GPU Type        | GPU Arch | GPUs | GPU RAM | vCPUs | RAM  | On-demand | Per-GPU | Spot | Name       |
|-----------------|----------|------|---------|-------|------|-----------|---------|------|------------|
| A100 (80 GB)    | Ampere   | 1    | 80      | 24    | 220  | $3.67     | $3.67   | $1.47| NC24ads A100 v4|
| A100 (80 GB)    | Ampere   | 2    | 160     | 48    | 440  | $7.35     | $3.67   | $2.94| NC48ads A100 v4|
| A100 (80 GB)    | Ampere   | 4    | 320     | 96    | 880  | $14.69    | $3.67   | $5.88| NC96ads A100 v4|
| A100 (80 GB)    | Ampere   | 8    | 640     | 96    | 1900 | $37.18    | $4.64   |       | ND96amsr A100 v4|
| V100 (16 GB)    | Volta    | 1    | 16      | 6     | 112  | $3.06     | $3.06   | $1.35| NC6s v3     |
| V100 (16 GB)    | Volta    | 2    | 32      | 12    | 224  | $6.12     | $3.06   | $2.7 | NC12s v3    |
| V100 (16 GB)    | Volta    | 4    | 64      | 24    | 448  | $12.24    | $3.06   | $5.4 | NC24s v3    |
| P100 (16 GB)    | Pascal   | 1    | 16      | 6     | 112  | $2.07     | $2.07   | $0.22| NC6s v2     |
| P100 (16 GB)    | Pascal   | 2    | 32      | 12    | 224  | $4.14     | $2.07   | $0.43| NC12s v2    |
| P100 (16 GB)    | Pascal   | 4    | 64      | 24    | 448  | $8.28     | $2.07   | $0.87| NC24s v2    |
| T4 (16 GB)      | Turing   | 1    | 16      | 4     | 28   | $0.53     | $0.53   | $0.21| NC4as T4 v3 |
| T4 (16 GB)      | Turing   | 1    | 16      | 8     | 56   | $0.75     | $0.75   | $0.3 | NC8as T4 v3 |
| T4 (16 GB)      | Turing   | 1    | 16      | 16    | 110  | $1.2      | $1.2    | $0.48| NC16as T4 v3|
| T4 (16 GB)      | Turing   | 4    | 64      | 64    | 440  | $4.35     | $1.09   | $1.73| NC64as T4 v3|
| K80 (12 GB)     | Kepler   | 1    | 12      | 6     | 56   | $0.9      | $0.9    | $0.09| NC6        |
| K80 (12 GB)     | Kepler   | 2    | 24      | 12    | 112  | $1.8      | $0.9    | $0.18| NC12       |
| K80 (12 GB)     | Kepler   | 4    | 48      | 24    | 224  | $3.6      | $0.9    | $0.36| NC24       |

# GCP
| GPU Type        | GPU Arch | GPUs | GPU RAM | vCPUs | RAM  | On-demand | Per-GPU | Spot | Name           |
|-----------------|----------|------|---------|-------|------|-----------|---------|------|----------------|
| A100 (40 GB)    | Ampere   | 1    | 40      | 12    | 85   | $3.67     | $3.67   | $1.1 | a2-highgpu-1g  |
| A100 (40 GB)    | Ampere   | 2    | 80      | 24    | 170  | $7.34     | $3.67   | $2.2 | a2-highgpu-2g  |
| A100 (40 GB)    | Ampere   | 4    | 160     | 48    | 340  | $14.68    | $3.67   | $4.41| a2-highgpu-4g  |
| A100 (40 GB)    | Ampere   | 8    | 320     | 96    | 680  | $29.36    | $3.67   | $8.81| a2-highgpu-8g  |
| A100 (40 GB)    | Ampere   | 16   | 640     | 96    | 1360 | $55.68    | $3.48   | $16.72| a2-megagpu-16g|
| V100 (16 GB)    | Volta    | 1    | 16      | 8     | 52   | $2.95     | $2.95   | $0.84| n1-highmem-8   |
| V100 (16 GB)    | Volta    | 2    | 32      | 16    | 104  | $5.91     | $2.95   | $1.68| n1-highmem-16  |
| V100 (16 GB)    | Volta    | 4    | 64      | 32    | 208  | $11.81    | $2.95   | $3.36| n1-highmem-32  |
| V100 (16 GB)    | Volta    | 8    | 128     | 64    | 416  | $23.63    | $2.95   | $6.72| n1-highmem-64  |
| P100 (16 GB)    | Pascal   | 1    | 16      | 8     | 52   | $1.93     | $1.93   | $0.53| n1-highmem-8   |
| P100 (16 GB)    | Pascal   | 2    | 32      | 16    | 104  | $3.87     | $1.93   | $1.06| n1-highmem-16  |
| P100 (16 GB)    | Pascal   | 4    | 64      | 32    | 208  | $7.73     | $1.93   | $2.12| n1-highmem-32  |
| K80 (12 GB)     | Kepler   | 1    | 12      | 8     | 52   | $0.92     | $0.92   | $0.14| n1-highmem-8   |
| K80 (12 GB)     | Kepler   | 2    | 24      | 16    | 104  | $1.85     | $0.92   | $0.28| n1-highmem-16  |
| K80 (12 GB)     | Kepler   | 4    | 48      | 32    | 208  | $3.69     | $0.92   | $0.56| n1-highmem-32  |
| K80 (12 GB)     | Kepler   | 8    | 96      | 64    | 416  | $7.39     | $0.92   | $1.12| n1-highmem-64  |
| T4 (16 GB)      | Turing   | 1    | 16      | 8     | 52   | $0.82     | $0.82   | $0.21| n1-highmem-8   |
| T4 (16 GB)      | Turing   | 2    | 32      | 16    | 104  | $1.65     | $0.82   | $0.42| n1-highmem-16  |
| T4 (16 GB)      | Turing   | 4    | 64      | 32    | 208  | $3.29     | $0.82   | $0.84| n1-highmem-32  |
| P4 (8 GB)       | Pascal   | 1    | 8       | 4     | 26   | $0.84     | $0.84   | $0.27| n1-highmem-4   |
| P4 (8 GB)	| Pascal	|2	|16	|8	|52	|1.67	|0.84	|0.54	|n1-highmem-8|
|P4 (8 GB)	| Pascal	|4	| 32	| 16	| 104	| 3.35	| 0.84	| 1.08	| n1-highmem-16
