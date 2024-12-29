# 🚀 AI Models & Tools Compendium 2024: Open Source & Beyond

> 📅 Last Updated: December 29, 2024
>
> 💡 A curated collection of AI models, prioritizing open-source solutions with commercial options included

## 📚 Table of Contents
- [Open Source Models](#open-source-models)
  - [Apache 2.0 Licensed](#apache-20-licensed)
    - [Falcon LLM](#falcon-llm)
    - [Mistral 7B](#mistral-7b)
    - [GPT-J & GPT-NeoX](#gpt-j--gpt-neox)
    - [RedPajama](#redpajama)
    - [Vicuna](#vicuna)
    - [Baichuan](#baichuan)
    - [Phoenix](#phoenix)
    - [Cerebras-GPT](#cerebras-gpt)
    - [OpenAssistant](#openassistant)
    - [Flan-T5](#flan-t5)
  - [MIT Licensed](#mit-licensed)
    - [Galactica](#galactica)
  - [OpenRAIL Licensed](#openrail-licensed)
    - [BLOOM](#bloom)
  - [Permissive Licensed](#permissive-licensed)
    - [Dolly](#dolly)
- [Commercial & Hybrid Models](#commercial--hybrid-models)
  - [Custom / Mixed License](#custom--mixed-license)
    - [LLaMA](#llama)
    - [Alpaca](#alpaca)
    - [ChatGLM](#chatglm)
    - [Qwen](#qwen)
    - [Yi-Lightning](#yi-lightning)
    - [DeepSeek](#deepseek)
- [Specialized Models](#specialized-models)
  - [Image Generation](#image-generation)
    - [Stable Diffusion](#stable-diffusion)
    - [DALL-E Mini](#dall-e-mini)
    - [DeepFloyd IF](#deepfloyd-if)
    - [Disco Diffusion](#disco-diffusion)
    - [Open Journey](#open-journey)
  - [Speech Models](#speech-models)
    - [Text-to-Speech](#text-to-speech)
      - [Coqui TTS](#coqui-tts)
      - [Mozilla TTS](#mozilla-tts)
      - [Piper](#piper)
    - [Speech-to-Text](#speech-to-text)
      - [Whisper](#whisper)
      - [DeepSpeech](#deepspeech)
      - [Kaldi](#kaldi)
- [Infrastructure & Platforms](#infrastructure--platforms)
  - [Cloud & Deployment](#cloud--deployment)
    - [Hugging Face](#hugging-face)
    - [RunPod](#runpod)
    - [Google Colab](#google-colab)
    - [InvokeAI](#invokeai)
    - [Replicate](#replicate)

## 🔓 Open Source Models

### Apache 2.0 Licensed
1. 🦅 Falcon LLM (TII) <span id="falcon-llm"></span>
   • License: Apache 2.0
   • Repository: [https://huggingface.co/tiiuae](https://huggingface.co/tiiuae)
   • Parameters: 40B/180B
   • Notes: High-performance NLP tasks

2. ⚡ Mistral 7B (Mistral AI) <span id="mistral-7b"></span>
   • License: Apache 2.0
   • Repository: [https://mistral.ai/](https://mistral.ai/)
   • Parameters: 7B
   • Notes: Dense LLM, efficient performance

3. 🤖 GPT-J (EleutherAI) <span id="gpt-j--gpt-neox"></span>
   • License: Apache 2.0
   • Repository: [https://github.com/kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax)
   • Parameters: 6B
   • Notes: Broad language tasks

4. 🧠 GPT-NeoX (EleutherAI) <span id="gpt-j--gpt-neox"></span>
   • License: Apache 2.0
   • Repository: [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)
   • Parameters: -
   • Notes: GPT-3 alternative

5. 🔄 RedPajama (Together) <span id="redpajama"></span>
   • License: Apache 2.0
   • Repository: [https://github.com/togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data)
   • Parameters: -
   • Notes: Democratizing access

6. 💬 Vicuna (LMSYS) <span id="vicuna"></span>
   • License: Apache 2.0
   • Repository: [https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat)
   • Parameters: 7B/13B
   • Notes: LLaMA-based conversational

7. 🌏 Baichuan (Baichuan AI) <span id="baichuan"></span>
   • License: Apache 2.0
   • Repository: [https://github.com/baichuan-inc/Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B)
   • Parameters: 13B
   • Notes: Chinese-focused NLP

8. 🎭 Phoenix (Phoenix AI) <span id="phoenix"></span>
   • License: Apache 2.0
   • Repository: [https://github.com/FreedomIntelligence/LLMZoo](https://github.com/FreedomIntelligence/LLMZoo)
   • Parameters: -
   • Notes: Bilingual reasoning

9. 🧮 Cerebras-GPT (Cerebras) <span id="cerebras-gpt"></span>
   • License: Apache 2.0
   • Repository: [https://huggingface.co/cerebras](https://huggingface.co/cerebras)
   • Parameters: -
   • Notes: Hardware-optimized

10. 🤝 OpenAssistant (LAION) <span id="openassistant"></span>
    • License: Apache 2.0
    • Repository: [https://github.com/LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
    • Parameters: -
    • Notes: Community-driven

11. 📚 Flan-T5 (Google) <span id="flan-t5"></span>
    • License: Apache 2.0
    • Repository: [https://github.com/google-research/t5x](https://github.com/google-research/t5x)
    • Parameters: -
    • Notes: Instruction-tuned

### MIT Licensed
1. 🧪 Galactica (Meta) <span id="galactica"></span>
   • License: MIT
   • Repository: [https://github.com/paperswithcode/galai](https://github.com/paperswithcode/galai)
   • Notes: Science-focused

### OpenRAIL Licensed
1. 🌺 BLOOM (BigScience) <span id="bloom"></span>
   • License: OpenRAIL
   • Repository: [https://huggingface.co/bigscience/bloom](https://huggingface.co/bigscience/bloom)
   • Parameters: 176B
   • Notes: Supports 46 natural & 13 programming languages (59 total?)

### Permissive Licensed
1. 🐑 Dolly (Databricks) <span id="dolly"></span>
   • License: Permissive
   • Repository: [https://github.com/databrickslabs/dolly](https://github.com/databrickslabs/dolly)
   • Notes: Fine-tuned from GPT-J/NeoX, open-source licensing

## 🔐 Commercial & Hybrid Models

### Custom / Mixed License
1. 🔥 LLaMA (Meta) <span id="llama"></span>
   • License: Custom (Research/Commercial)
   • Repository: [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)
   • Parameters: 7B–70B
   • Notes: Multilingual, flexible fine-tuning

2. 🦙 Alpaca (Stanford) <span id="alpaca"></span>
   • License: Custom (Research)
   • Repository: [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
   • Parameters: -
   • Notes: Fine-tuned from LLaMA for instruction tasks

3. 💬 ChatGLM (THUDM) <span id="chatglm"></span>
   • License: Commercial allowed
   • Repository: [https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
   • Parameters: 6B
   • Notes: CN/EN bilingual

4. 🌟 Qwen (Alibaba) <span id="qwen"></span>
   • License: Partial commercial
   • Repository: [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
   • Parameters: 7B-110B
   • Notes: MoE variants

5. ⚡ Yi-Lightning (Yi AI) <span id="yi-lightning"></span>
   • License: Proprietary
   • Repository: [https://yi.ai/](https://yi.ai/)
   • Notes: Cost-efficient performance

6. 🔍 DeepSeek (DeepSeek AI) <span id="deepseek"></span>
   • License: Mixed commercial
   • Repository: [https://deepseek.ai/](https://deepseek.ai/)
   • Notes: Large context window

## 🎯 Specialized Models

### 🎨 Image Generation
1. Stable Diffusion (Stability AI) <span id="stable-diffusion"></span>
   • License: CreativeML
   • Repository: [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
   • Notes: Text-to-image diffusion

2. DALL-E Mini (Boris Dayma) <span id="dall-e-mini"></span>
   • License: Apache 2.0
   • Repository: [https://github.com/borisdayma/dalle-mini](https://github.com/borisdayma/dalle-mini)
   • Notes: Accessible image synthesis

3. DeepFloyd IF (DeepFloyd) <span id="deepfloyd-if"></span>
   • License: Mixed
   • Repository: [https://github.com/deepfloyd/IF](https://github.com/deepfloyd/IF)
   • Notes: High-fidelity generation

4. Disco Diffusion (Colab) <span id="disco-diffusion"></span>
   • License: MIT
   • Repository: [https://github.com/alembics/disco-diffusion](https://github.com/alembics/disco-diffusion)
   • Notes: Artistic creation

5. Open Journey (PromptHero) <span id="open-journey"></span>
   • License: CreativeML
   • Repository: [https://github.com/prompthero/openjourney](https://github.com/prompthero/openjourney)
   • Notes: Style-focused

### 🗣️ Speech Models
1. Coqui TTS (Coqui) <span id="coqui-tts"></span>
   • Type: TTS
   • License: MPL 2.0
   • Repository: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
   • Notes: Multilingual

2. Mozilla TTS (Mozilla) <span id="mozilla-tts"></span>
   • Type: TTS
   • License: MPL 2.0
   • Repository: [https://github.com/mozilla/TTS](https://github.com/mozilla/TTS)
   • Notes: Natural voices

3. Piper (Rhasspy) <span id="piper"></span>
   • Type: TTS
   • License: MIT
   • Repository: [https://github.com/rhasspy/piper](https://github.com/rhasspy/piper)
   • Notes: Edge deployment

4. Whisper (OpenAI) <span id="whisper"></span>
   • Type: STT
   • License: MIT
   • Repository: [https://github.com/openai/whisper](https://github.com/openai/whisper)
   • Notes: Recognition

5. DeepSpeech (Mozilla) <span id="deepspeech"></span>
   • Type: STT
   • License: MPL 2.0
   • Repository: [https://github.com/mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech)
   • Notes: RNN-based

6. Kaldi (Kaldi) <span id="kaldi"></span>
   • Type: STT
   • License: Apache 2.0
   • Repository: [https://github.com/kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi)
   • Notes: Research toolkit

## ⚙️ Infrastructure & Platforms

### Cloud & Deployment
- 🤗 **Hugging Face** - Model repository & deployment ([https://huggingface.co/](https://huggingface.co/))
- ☁️ **RunPod** - GPU infrastructure ([https://runpod.io/](https://runpod.io/))
- 📓 **Google Colab** - Notebook environment ([https://colab.research.google.com/](https://colab.research.google.com/))
- 🎨 **InvokeAI** - Stable Diffusion platform ([https://invoke.ai/](https://invoke.ai/))
- 🔄 **Replicate** - API deployment ([https://replicate.com/](https://replicate.com/))

## 👥 How to Contribute

We welcome contributions! See our [contribution guidelines](CONTRIBUTING.md) for:
- Adding new models
- Updating existing information
- Reporting outdated links
- Suggesting improvements

Your contributions help keep this resource accurate and up-to-date.

---
> 📋 **Note**: All repository links are preserved for direct access
> ⚖️ **License Notice**: Always verify current license terms before deployment
> 🔍 **Navigation Tip**: Use the Table of Contents to quickly jump to specific models
