# LLMFlows - Simple, Explicit, and Transparent LLM Apps

<p align="center">
  <img style="width: 80%" src="docs/llmflows_last_logo.png" />
</p>

[![Twitter](https://img.shields.io/twitter/follow/LLMFlows?style=social)](https://twitter.com/LLMFlows)
![Pylint workflow](https://github.com/stoyan-stoyanov/llmflow/actions/workflows/pylint.yml/badge.svg)
![License](https://img.shields.io/github/license/stoyan-stoyanov/llmflow)
![PyPi](https://img.shields.io/pypi/v/llmflows)
![Stars](https://img.shields.io/github/stars/stoyan-stoyanov/llmflow?style=social)
![Release date](https://img.shields.io/github/release-date/stoyan-stoyanov/llmflow?style=social)

Documentation: [https://readthedocs.org](https://readthedocs.org/)<br/>
PyPI: [https://pypi.org/project/llmflows](https://pypi.org/project/llmflows/)</br>
Twitter: [https://twitter.com/LLMFlows](https://twitter.com/LLMFlows)<br/>
Substack: [https://llmflows.substack.com](https://llmflows.substack.com/)<br/>

## About
LLMFlows is a framework for building simple, explicit, and transparent LLM applications.

## Installation
```
pip install llmflows
```

## Philosophy

### Simple
Our goal is to build a simple, well-documented framework with minimal abstractions that 
allow users to build flexible LLM-powered apps without compromising on capabilities.

### Explicit
We want to create an explicit API enabling users to write clean and readable code while 
easily creating complex flows of LLMs interacting with each other.

### Transparent
We aim to help users have full transparency on their LLM-powered apps by providing 
traceable flows and complete information for each app component, making it easy to 
monitor, maintain, and debug.


## Usage
You can quickly build a simple application with LLMFlows' `LLM` and `PrompTemplate` 
classes:

```python

from llmflows.llms.openai import OpenAI
from llmflows.prompts.prompt_template import PromptTemplate

prompt_template = PromptTemplate(
   prompt="Generate a title for a 90s hip-hop song about {topic}."
)
llm_prompt = prompt_template.get_prompt(topic="friendship")

llm = OpenAI()
song_title = llm.generate(llm_prompt)

```

However, real-world applications are often more complex and have dependencies between 
prompts and LLM calls. For example:

![Complex flow](docs/complex_flow.png)

You can build such flow using the `Flow` and `Flowstep` classes. LLMFlows will figure 
out the dependencies and make sure each flowstep runs only when all its dependencies 
are met:

```python
from llmflows.flows import Flow, FlowStep
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

# Create prompt templates
title_template = PromptTemplate("What is a good title of a movie about {topic}?")
song_template = PromptTemplate(
    "What is a good song title of a soundtrack for a movie called {movie_title}?"
)
characters_template = PromptTemplate(
    "What are two main characters for a movie called {movie_title}?"
)
lyrics_template = PromptTemplate(
    "Write lyrics of a movie song called {song_title}. The main characters are "
    "{main_characters}"
)

# Create flowsteps
flowstep1 = FlowStep(
    name="Flowstep 1",
    llm=OpenAI(),
    prompt_template=title_template,
    output_key="movie_title",
)

flowstep2 = FlowStep(
    name="Flowstep 2",
    llm=OpenAI(),
    prompt_template=song_template,
    output_key="song_title",
)

flowstep3 = FlowStep(
    name="Flowstep 3",
    llm=OpenAI(),
    prompt_template=characters_template,
    output_key="main_characters",
)

flowstep4 = FlowStep(
    name="Flowstep 4",
    llm=OpenAI(),
    prompt_template=lyrics_template,
    output_key="song_lyrics",
)

# Connect flowsteps
flowstep1.connect(flowstep2, flowstep3, flowstep4)
flowstep2.connect(flowstep4)
flowstep3.connect(flowstep4)

# Create and run Flow
soundtrack_flow = Flow(flowstep1)
results = soundtrack_flow.start(topic="friendship", verbose=True)
```

In addition, LLMFlows provides async classes to improve the runtime of any complex flow 
by running flow steps that already have all their required inputs in parallel.
Check our documentation for more examples, such as creating question-answering apps and 
web applications with Flask and FastAPI.

## Features

### LLM Integration
- Utilize LLMs such as OpenAI's GPT-3 for generating natural language text.
- Configure LLM classes with specific models, settings, and parameters.
- Use the `generate()` method to generate text based on prompts.

### Prompt Templates
- Create dynamic prompts using Prompt Templates.
- Define variables within the prompts that can be filled in later.
- Generate prompt strings by providing the required variables.

### Flows and FlowSteps
- Structure your LLM applications using Flows and FlowSteps.
- Create a directed acyclic graph (DAG) of flow steps to define the order of execution.
- Connect flow steps to pass outputs as inputs to subsequent steps.
- Execute the flow using the `start()` method to perform the defined steps.

### VectorStore Integration
- Leverage vector databases with the VectorStoreFlowStep for efficient storage and 
retrieval of vector embeddings.
- Connect to vector databases such as Pinecone for scalable storage and querying of 
  vectors.

### FunctionalFlowStep
- Incorporate custom string manipulation functions into flows.
- Apply functions to manipulate text without relying on LLM calls.

### FunctionalCallback
- Execute callback functions at different stages within a flow step for enhanced 
customization and integration capabilities.
- Utilize callbacks for logging, tracing, or other custom requirements.

## Features

### LLMs
- Utilize LLMs such as OpenAI's ChatGPT to generate natural language text.
- Configure LLM classes easily, choosing specific models, parameters, and settings.
- Benefit from automatic retries when model calls fail, ensuring reliable LLM interactions.

### Prompt Templates
- Create dynamic prompts using Prompt Templates, providing flexible and customizable text generation.
- Define variables within prompts to generate prompt strings tailored to specific inputs.

### Flows and FlowSteps
- Structure LLM applications using Flows and FlowSteps, providing a clear and organized framework for executing LLM interactions.
- Connect flow steps to pass outputs as inputs, facilitating seamless data flow and maintaining a transparent LLM pipeline.
- Leverage Async Flows to run LLMs in parallel when all their inputs are available, optimizing performance and efficiency.
- Incorporate custom string manipulation functions directly into flows, allowing specialized text transformations without relying solely on LLM calls.

### VectorStore Integration
- Integrate with vector databases like Pinecone using the VectorStoreFlowStep, empowering efficient and scalable storage and retrieval of vector embeddings.
- Leverage vector databases for seamless storage and querying of vectors, enabling straightforward integration with LLM-powered applications.

### Callbacks
- Execute callback functions at different stages within flow steps, enabling enhanced customization, logging, tracing, or other specific integrations.
- Utilize callbacks to comprehensively control and monitor LLM-powered apps, ensuring clear visibility into the execution process.

## FAQ

### **How is this different than langchain?**
Langchain is a great library, and LLMFlows has undoubtedly been inspired by it. 
However, our philosophy is a bit different. Langchian has a "chain for everything" 
philosophy and provides many classes with multiple LLM calls, logic, and built-in 
default prompts. While this is great for beginners and default use cases, we feel this 
can be overwhelming if users want to do anything "out of the ordinary." In contrast, 
we are focusing on providing as few building blocks as possible and having an 
easy-to-understand API while matching (and in some cases exceeding) the capabilities 
of langchain.

### **You only have OpenAI wrappers, but I want to use AcmeLLM.**
We decided to release the library initially supporting only OpenAI LLMs, but we have a 
roadmap and will slowly add new wrappers around the most popular models. If you are 
willing to spend some time, we are looking for contributors and maintainers.

### **You only support Pinecone. Do you have plans to extend the list?**
Yes! Over time, we will also add Chroma, Weaviate, Redis, Elastic Search, and other 
popular solutions. If you want to help us out, check out our contribution section.

### **Why can't I find any info related to document loaders?**
For the time being, we have decided not to implement document loaders for a few reasons:

1. plenty of capable libraries like Llama-index and langchain have tons of loaders.
2. We think mixing document loaders with LLM and prompt management libraries is awkward 
3. since document loading usually happens in separate pipelines and is not part of the 
4. LLM-powered app.
5. Real-life documents are messy. In our experience, no matter how many loaders are out 
6. there, they will never cover all the specific use cases.

While we will not invest time into document loaders, we might change direction if we 
get significant interest and contributors.

## License
LLMFlows is covered by the MIT license. For more information, check `LICENCE.md`.

## Contributing
Thank you for spending the time to read our README! If you like what you saw and are 
considering contributing, please check `CONTRIBUTING.md`
