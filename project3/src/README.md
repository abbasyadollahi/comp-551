# Modified MNIST

*Part of Project 3 of COMP 551 Applied Machine Learning - McGill University*  
*Members: Le Nhat Hung, Yadollahi, Alex Harris*

## Prerequisites

### Running Google Colab locally *(optional)*

1. Install and enable the jupyter_http_over_ws jupyter extension (one-time)
    ```
    pip install jupyter_http_over_ws
    jupyter serverextension enable --py jupyter_http_over_ws
    ```

2. Start server and authenticate
    ```
    jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
    ```
