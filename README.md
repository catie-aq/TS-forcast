# TS-forcast

# Time series foundation models

| Papers |  | Limitations |
| --- | --- | --- |
| TimeGPT [https://arxiv.org/abs/2310.03589](https://arxiv.org/abs/2310.03589) | adapt the techniques and architecture behind LLMs to the field of forecasting, successfully building the first time series foundation model capable of zero-shot inference. | Not open source. The paper remains vague in many important areas.
do not know what datasets were used to train and test the model, so we cannot really verify the performance results of TimeGPT |
| TimeFM [https://arxiv.org/html/2310.10688v2](https://arxiv.org/html/2310.10688v2) | pretraining a patched-decoder style attention model on a large time-series corpus comprising both real-world and synthetic datasets | Not open source |
| (Jin et al., 2023), LLM4TS (Chang et al., 2023), GPT2(6)
(Zhou et al., 2023a), UniTime (Liu et al., 2023), and TEMPO
(Anonymous, 2024) |  freeze LLM encoder backbones while
simultaneously fine-tuning/adapting the input and distribution heads for forecasting |  |
| [https://time-series-foundation-models.github.io/lag-llama.pdf](https://time-series-foundation-models.github.io/lag-llama.pdf) | foundation model for univariate **probabilistic time series** forecasting based on a decoder-only transformer architecture that uses lags as covariates | - Univariate time series.  the downside to using lagged features in tokenization
is that it requires an L-sized or larger context window |
| [https://arxiv.org/pdf/2310.10196.pdf](https://arxiv.org/pdf/2310.10196.pdf) | Large Models for Time Series and
Spatio-Temporal Data: A Survey and Outlook |  |
| [**Chronos: Learning the Language of Time Series**](https://arxiv.org/pdf/2403.07815.pdf) | framework for pretrained **probabilistic time
series models.** Chronos tokenizes time series values using scaling and quantization into
a fixed vocabulary and trains existing transformer-based language model architectures on
these tokenized time series via the cross-entropy loss | the prediction range is restricted between [c1, cB],
making it theoretically infeasible to model time series with a strong trend.  The model does not allow external information such as static (product brand, color, etc.) or dynamic (product price, macroeconomic data, etc.) covariates. Also, it treats each time series as a simple sequence without time or frequency information (hourly, daily, weekly, or monthly data), which might become a disadvantage when modeling seasonality. Another limitation is the fact that it is a univariate model only. Additionally, it can only forecast one time series at a time, which does not allow for modeling dependencies between time series. |
|  PromptCast (Xue &
Salim, 2023) | leverages pretrained LLMs for forecasting by transforming the time series data into text-based
input and output pairs and reformulating the forecasting problem as a question answering task | requires dataset-specific templates for converting numerical data to text prompts |
| LLMTime (Gruver et al., 2023) | encodes real-valued data as a string of digits after fixing
the numerical precision and scaling the data appropriately. Once encoded as strings, forecasts are obtained
in a zero-shot setting from pretrained LLMs such as GPT-3  and Llama 2  | the use of such compute-hungry models hampers the scalability and practical
utility of LLMTim |
| GPT4TS (Zhou et al. (2023a) | unified one-fits-all model for different time series analysis tasks
by using a pretrained GPT-2 model as a backbone and only fine-tune the positional
embeddings and the parameters of the layer normalization for each individual task. Instead of using tokenized
input, they directly feed the model with patch embeddings, similar to |  |
|  PatchTST (Nie et al., 2023) [https://arxiv.org/pdf/2211.14730.pdf](https://arxiv.org/pdf/2211.14730.pdf) | channel-independence to forecast multivariate time series. PatchTST makes use of patching to extract local semantic information in time series.
Enabling the model to attend to longer historical information. | higher computational and memory requirements |
|  Time-LLM (Jin et al., 2024) | align embeddings of time series patches with text prototypes, and prompting the (frozen) LLM with these aligned
embeddings and a natural language prefix describing the task | require in-domain training or fine-tuning, i.e., they are fine-tuned and tested on each dataset separately |
| (Rasul et al., 2023; Goswami et al., 2024; Das et al., 2023; Woo et al., 2024 | develop zero-shot forecasting models by pretraining transformer-based architectures on a large corpus
of time series data. These works operate on the real values of the time series and include time-seriesspecific designs such as time features, lags, patching, and real-valued distribution heads, among others |  |
| Zhou et al. (2023),  | Studied general purpose
models applicable across time series tasks including imputation, forecasting, classification and anomaly
detection |  |
| SimMTM (Dong et al., 2023)  | masked pretraining framework for time series
which learns general time series representations that are then used for forecasting and classification via
fine-tuning. |  |
| MOIRAI [https://arxiv.org/pdf/2402.02592.pdf](https://arxiv.org/pdf/2402.02592.pdf) | Masked EncOder-based
UnIveRsAl TIme Series Forecasting Transformer.   Handle all kinds of data frequencies (hourly, daily, weekly, etc);
Accommodate any number and types of covariates, whether they are unknown in the future or known;
Generate a probabilistic forecast using a flexible distribution that can be adapted to several cases.                  |  |
| VISIONTS (2024)   [2408.17253](https://arxiv.org/pdf/2408.17253) | Formulation of TS as an image reconstruction task. Available code https://github.com/Keytoyze/VisionTS |  |

TimeGPT [4] was one of the first foundation models developed for forecasting by Nixtla. Following it, other companies entered the race and developed new models, such as MOIRAI [5] from Salesforce, Lag-Llama [6] from Meta, or TimesFM [7] from Google. More recently, Amazon joined them and developed Chronos [8], a foundational model for time series based on language model architectures.

stats models implementation: [https://github.com/liamarguedas/walmart-sales-forecast/blob/main/Walmart-Sales-Forecast.ipynb](https://github.com/liamarguedas/walmart-sales-forecast/blob/main/Walmart-Sales-Forecast.ipynb)

Comparing chronos with stats models: [https://github.com/Nixtla/nixtla/blob/main/experiments/amazon-chronos/src/statsforecast_pipeline.py](https://github.com/Nixtla/nixtla/blob/main/experiments/amazon-chronos/src/statsforecast_pipeline.py)