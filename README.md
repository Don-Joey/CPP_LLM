
<table>
<tr>
  <td><img src="images/logo.jpg" alt="Logo" width="175"></td>
  <td><h1> Collaborative Performance Prediction for Large Language Models</h1><p>Collabrative filtering method based on collaborative performance score matrix, model factors and task factors.</p></td>
</tr>
</table>

## Description
This project introduces the Collaborative Performance Prediction (CPP) framework, which enhances the predictability of large language model (LLM) performance across various tasks. It leverages historical performance data and design factors for both models and tasks to enhance prediction accuracy, offering significant improvements over traditional scaling laws. Our framework consists of two components: 1. collaborative performance data and 2) collaborative prediction methods, for instance, Matrix Factorization, Neural Collaborative Filtering and so on. We anticipate that an accurate score can be predicted based on the historical performance of various models on downstream tasks.

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

### preprocess.ipynb
- **Description**: convert all tables in raw_data to the overall table all_benchmark_score.csv and all_benchmark_rank.csv.

### matric_factorization_rank.ipynb
- **Description**: train and predict rank.

### matric_factorization_score.ipynb
- **Description**: train and predict original score, then convert the score to rank.

### matric_factorization_norm_score.ipynb
- **Description**: train and predict normalized score, then convert the score to rank.
