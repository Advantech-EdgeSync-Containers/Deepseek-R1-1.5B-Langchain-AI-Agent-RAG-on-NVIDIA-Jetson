# Model Customization & Environment Configuration

## Model Customization 
If needed, the Ollama model can be customized as per the requirements. Follow this for customizing the model to achieve expected inference results by tweaking the following parameters using Modelfile: 

| Parameter      | Description                                                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------|
| SYSTEM         | Defines the default behavior or personality of the model. Acts like a system prompt for alignment (e.g., "You are a coding assistant."). |
| TEMPLATE       | Defines the prompt structure used (e.g., chatml, llama, alpaca, etc.) to format user/assistant messages.                                 |
| TEMPERATURE    | Controls randomness. Lower = more deterministic; higher = more creative/unpredictable. Range: 0.0 – 2.0                                  |
| REPEAT_PENALTY | Penalizes repetition in output. Values greater than 1.0 reduce repeated phrases. Default is around 1.1                                               |
| TOP_P          | Enables nucleus sampling—limits token selection to a cumulative probability (e.g., 0.95). Helps control diversity                      |
| TOP_K          | Limits sampling to the top K most probable next tokens (e.g., 50). Lower values = more focused output                                    |

For other parameters refer to this official documentation from [**Ollama**](https://github.com/ollama/ollama/blob/main/docs/modelfile.md). These are the steps to create a model using Modelfile. The above parameters can be customized by either using Modelfile as below ```.env``` or by using the Modelfile. In case of customization done by Modelfile, do replace the MODEL_NAME in ```.env``` the file with the new model name.

- Create a Modelfile

    ```
    touch Modelfile
    ```

- Example Modelfile
  For details about these parameters, take a look at the ```.env``` file or Ollama documentation as mentioned above.
  ```
  # Mention the model to be used, it could be any Ollama model 
  # or local .gguf model (placed at same path as Modelfile)
  FROM deepseek-r1:1.5b 

  # System prompt to guide model behavior 

  SYSTEM "You are a helpful assistant. Answer concisely and accurately." 

  # Use a predefined prompt formatting template 

  TEMPLATE "chatml" 

  # Adjust model inference parameters 

  PARAMETER temperature 0.5

  PARAMETER repeat_penalty 1.1 

  PARAMETER top_p 0.65

  PARAMETER top_k 40 
  ```

- Build the model using Ollama. 
  ```
  ollama create deepseek-r1-1.5b-custom -f Modelfile 
  ```

- Run the model directly using the following command: 
  ```
  ollama run deepseek-r1-1.5b-custom --verbose
  ```  

The custom model can also be accessed directly from OpenWebUI; on the top left, go to the Select Model dropdown, select the custom model, and start using it. 


## Environment Configuration

The `.env` file allows you to customize the runtime behavior of the container using environment variables.

### Key Environment Variables
``` bash
# --- Model Settings ---
# Any other ollama or custom built model can be used here if needed
MODEL_NAME=deepseek-r1:1.5b

# --- Ollama Settings ---
COMPOSE_FILE_PATH=./docker-compose.yml

# The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
TEMPERATURE=0.5

# Sets the size of the context window used to generate the next token. (Default: 2048)
#NUM_CTX=

# The number of GPUs to use. On macOS it defaults to 1 to enable metal support, 0 to disable.
#NUM_GPU=

# Sets the number of threads to use during computation. By default, Ollama will detect this for optimal performance.
# It is recommended to set this value to the number of physical
# CPU cores your system has (as opposed to the logical number of cores).
#NUM_THREAD=

# System prompt (overrides what is defined in the Modelfile)
#SYSTEM=

# How long the model will stay loaded into memory.
# The parameter (Default: 5 minutes) can be set to:
#  1. a duration string in Golang (such as "10m" or "24h");
#  2. a number in seconds (such as 3600);
#  3. any negative number which will keep the model loaded in memory (e.g. -1 or "-1m");
#  4. 0 which will unload the model immediately after generating a response;
#  See the [Ollama documents](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)"""
#KEEP_ALIVE=

# Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
# will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
#REPEAT_PENALTY=

# Reduces the probability of generating nonsense. A higher value (e.g. 100)
# will give more diverse answers, while a lower value (e.g. 10)
# will be more conservative. (Default: 40)
TOP_K=40

# Works together with top-k. A higher value (e.g., 0.95) will lead
# to more diverse text, while a lower value (e.g., 0.5) will
# generate more focused and conservative text. (Default: 0.9)
TOP_P=0.65

# Full prompt or prompt template (overrides what is defined in the Modelfile)
#TEMPLATE=

#Other similar parameters can be added which are supported by Ollama, make sure to add them also in Ollama instantiation within llm_loader.py file

# --- LANGCHAIN FASTAPI Settings ---
FASTAPI_LANGCHAIN_PORT=8000
OLLAMA_API_BASE=http://localhost:11434

# --- OPENWEBUI Settings ---
OPENWEBUI_PORT=3000
OPENAI_API_LANGCHAIN_BASE=http://localhost:8000

 
```

## Ollama External Access
- Access Ollama on all network interfaces, i.e., outside devices also
  ```
  # Export OLLAMA_HOST as following
  export OLLAMA_HOST=0.0.0.0

  # Kill existing Ollama service
  kill $(cat ollama.pid)

  # Confirm that there is no such process named as Ollama, otherwise kill using its pid
  ps -eaf
  
  # Run `start_services.sh script`
  ./start_services.sh
  ```

  ```
- Now verify that Ollama should be accessible on the following address on a browser externally
  ```
    http://<device_ip>:11434
  ```
  It should display – ‘Ollama is running’ 

  ![ollama-status](..%2Fdata%2Fimages%2Follama-status.png)


## Ollama Model Memory Usage Optimization

### Enabling Flash Attention & KV Cache Quantization

Combining KV Cache Quantization and Flash Attention often results in better memory efficiency and performance. Different models behave differently for these settings, it is advised to test the accuracy after settings these for running intended models.


- Stop existing Ollama service 
- Export following environment variables :
  ```
  export OLLAMA_FLASH_ATTENTION=1 # this enables flash attention
  export OLLAMA_KV_CACHE_TYPE=q4_0 # f16 is by default, q8_0 reduces cache usage to half of f16, and q4_0 to 1/4th of usage in f16
  ```
- Start the Ollama service. 
- Check memory utilization by ollama logs; kv-cache size would be reduced. 
- Before Optimization
![kv-cache-before](..%2Fdata%2Fimages%2Fkvcache-before.png)

- After Optimization
![kv-cache-before](..%2Fdata%2Fimages%2Fkvcache-after.png)