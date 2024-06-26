Over the past year, I have found some great learning resources and insights relevant to Deep Learning. I keep track of things that I come across, and it can be helpful to you. I haven't gone through much of this, primarily due to procrastination, but it can help you start your journey, and I hope you make a lot better use of it than I do.  

This will mostly be in the context of Deep Learning. My goal is to leave you with some idea of the research areas and architecture developed in Deep Learning, along with the resources for working and learning about them.

We will first peek into the **different domains**, and then I will link some **excellent resources** I have come across. Please pardon my grammar; I wrote this in a limited time.

### **Broad Topics**

Computer Vision and Generative Modeling

Natural Language Processing

Speech Recognition

Reinforcement Learning

Interpretability - Reverse Engineering Neural Networks

Deep Learning Frameworks

Efficient Deep Learning

### **Computer Vision and Generative Modeling**

Computer Vision mainly deals with image modality and how we can learn features in a particular image and use these features to train a deep neural network for a CV task. The features previously were hand-engineered but can be learned by training a special class of Neural Networks called **Convolution Neural Networks (CNN)**. CNN dominates most of the computer vision domain. They are very good at learning the hierarchical composition of an image. The significant tasks involved here are **classification** and **segmentation**. Again, they are broad classes that can be more specific in pose estimation, face recognition, super-resolution, etc.

One particular domain within this modality that excites me the most is generative modeling. Generative modeling deals with a particular range of tasks that don't have any labels. For example, given a particular class of images, you train the model to learn its distribution and generate more similar images. **Generative Adversarial Network** (**GANs**) have been one of the best pieces of research which have evolved in this area. They are based on generator and discriminator structures. Previously, there were **autoencoders (VAEs),** which used encoder-decoder pipelines**.** They are still relevant in some contexts. 

Recently, **Diffusion Models** have been making an impact in the world of generative modeling. They are adapted from a critical idea in Thermodynamics. They are being looked at as the successor to GANs. If you have used **Stable Diffusion** or **DALL-E**, they use diffusion models on their backend. Along with diffusion models, a special class of models called Transformers(a groundbreaking work in NLP that gave us **ChatGPT**) is also being applied in Computer Vision. They are primary models in Natural Language Processing, but have also been used recently here and are known as **Vision Transformers**. **Open AI's CLIP** is one of the examples. 

Another exciting domain that I think not many know about is **Adversarial Learning**. Adversarial machine learning studies the attacks on machine learning algorithms and the defenses against such attacks. It's like hacking a neural network and learning about its potential parameters. 

**Ps**:  In **the 10th Inter IIT Tech Meet**, **IIT BHU** bagged **a Silver** medal in a problem presented by **Bosch** on Adversarial Learning.

### **Natural Language Processing**

If you ask me which domain in ML people are most crazy about, it's NLP. Every week hundreds of papers, work, deployment across the Conferences, and Github. It deals with text modality and how to compose text and train a network to learn about language and communication. Language is a fundamental way of expressing our thoughts and communicating with others, and if you can train a model on a massive chunk of text data, you come up with **ChatGPT.** 

The most daunting task here is representing words so a model can understand them. With images, it's easier to express pixel values across **R, G, and B** channels with values between ** 0-255.** But with words, it's much more complicated. It's a fun topic that I can write about a lot. We take what is called a **token**. A token is a segment representing a complete word or a subword. For example, **playing** can be described with **play + ing**; these two are individual tokens from a word. After **tokenization**, we assign each token a position according to our vocabulary and then generate **rich embeddings** to represent them. The most important thing is how we can generate those rich embeddings.

Previously, these embeddings were generated using an algorithm called **word2vec**. The embeddings were then fed into feed-forward networks to perform an NLP task. General NLP tasks include **translation, sentiment classification, language modeling, part-of-speech tagging**, etc. 

Sentences are sequential and are composed of words related to each other. Thus a different class of models called **Recurrent Neural Networks** were invented to learn about this sequential composition. Later, more research went into improving RNNs leading to **LSTMs and GRUs**. But RNNs have their own drawbacks and, today, we use **Transformers** for NLP tasks. Transformers are huge deep neural networks consisting of billions of parameters and trained on huge chunks of data resulting in **Large Language Models (LLMs)**. LLMs today dominate the NLP industry with a lot of startups and big names such as OpenAI, Google, and Microsoft betting their money on it.

**Hugging Face** is the place to go for using different LLMs and Foundation Models. Its a hub where researchers and labs deploy there model for public use. You can load any model and use it or can finetune (if you have compute). 

One thing to note is training such models require a lot of compute (**GPUs or TPU**) which are expensive. So if you are from Electronics you can look into the direction of building GPUs. **"When everyone is digging for gold, sell the shovels".** 

### **Speech Recognition** 

As the name suggests, the domain involves the work of transcribing audio and understanding them. As simple as it may sound, trust me, it's not. Many people don't know but after the 2000s around when Deep Learning died, a surge again in the use of neural networks came in the application of speech recognition. The major tasks involve **speech recognition, speech translation, speech diarization** (detecting different speakers in audio and who said what), and **speech correction**. It's an interesting field where architectures such as **RNNs or Transformers** are dominant. First, a spectrogram is generated from audio signals and then convolution and recurrent networks are used to perform the tasks.

Different libraries such as librosa can be used for generating audio features. Libraries such as **ESPNet, NeMo, and Pyannote** have a lot of models and tools for doing these tasks. OpenAI released its model named **Whisper** for this. Conversational AI is an interesting area to work in. I have recently seen some papers that are proposing the use of **LLMs in speech recognition** so if you are into LLMs this is a good place for application.

### **Reinforcement Learning**

Ah, the most intriguing and tough of them all. One of my senior and good friend often quotes life in terms of **Reinforcement Learning** and this is why I feel it's an important field. It tries to use the same pattern for learning as we all do. Using a **reward** for teaching a model. The most natural way of learning. Some interesting applications of RL can majorly be seen in training models to play **arcade games and Robotics Controls** . If you are into Robotics, you should look into RL. Another evolving direction is **Multi-agent RL** (MARL). It involves building multiple agents that interact with each other to either **compete or cooperate** for a particular task.

There is so much that can be said about the field but, I will leave it to you. One really interesting application of RL has been in ChatGPT. Reward Models are being used to **align LLMs** for generating human-preferred answers (**RLHF**). There are some really interesting applications of RL and I think it still is under explored.

[Microsoft conducts Reinforcement Learning Open Source](https://www.microsoft.com/en-us/research/academic-program/rl-open-source-fest/) (RLOS) Fest each year, where a handful of students are selected from across the world and get a chance to work at **Microsoft Research NYC** in the domain of RL. 

**Ps**: One of our **IIT BHU** senior got selected for **RLOS**.

### **Interpretability / Reverse Engineering**

Not many know about this field but it is one of the most interesting and developing field out there that demands low resources. Mechanistic Interpretability is all about reverse engineering neural networks and gaining an in-depth understanding of how they work or what they learn. Chrish Olah and Neel Nanda have been key contributer and their resources are really great for starting out. I will link them here for you.

[ZoomIn](https://distill.pub/2020/circuits/zoom-in/)

[Mechanistic Interpretability](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

### **Deep Learning Frameworks**

Okay, this is not an area of research or something, but I think some people overlook them. Deep Learning Frameworks are tool kits that are used to build and train neural networks. They provide you with hundreds of functionality to use. Neural Networks are built of **Tensors**. I like to think of Tensors as the fundamental unit of Deep Learning. Tensors are just like arrays but with more functionalities. Some famous frameworks included **PyTorch, Tensorflow, and JAX**. I personally use PyTroch, it is easy to learn and has Tensor functions quite similar to NumPy functions. In fact, you can build a small framework similar to **Torch in Numpy**. It's a fun project.

**Ps: Read Chapter 13 of Grokking Deep Learning.** 

A library I would like to refer to here is **Einops.** Einops is really handy for writing these neural networks. It allows you to perform **matrix multiplication, and tensor manipulation** in a more expressing way and is compatible with any deep learning framework.

### **Efficient Deep Learning**

This is focused on making neural networks efficient. LLMs or large CNN-based models require compute resources for training which are not always feasible. How can we make better use of **less compute for** training is what we try to answer here. It's an important and interesting field to work in and is mathematically rigorous. Some key techniques involve:

1. Quantization
2. Pruning
3. Distillation
4. Rank Decomposition
5. Neural Architecture Search
6. Mixture of Experts

**Efficient Finetuning of LLMs** is really being explored recently with techniques such as **LoRA**. Some useful libraries are **Accelerate and DeepSpeed**. Again this is compute demanding.

I have talked about different domains or important tools, this is a small intro-type thing just to let you hover around and leave you with some key ideas. Now I'll list some resources to help you learn about all these things and work on them.

## **Resources** 

Lectures 

Readings

Blogs

Github Repos

Competitions

Computes

Communities

Inter IIT 

### **Lectures**

People who have made a great impact in their respective area of research literally put their lectures on YouTube for you to learn.

1. Stanford Courses: [Cs231n](http://cs231n.stanford.edu/) (CV), [Cs224n](https://web.stanford.edu/class/cs224n/) (NLP),[ Cs234](https://web.stanford.edu/class/cs234/) (RL)
2. [NYU DL Course](https://atcold.github.io/NYU-DLSP20/) : Yann LeCun invented CNNs 
3. [Karapathy's Zero to Hero](https://karpathy.ai/zero-to-hero.html) : Karapathy is a legend
4. [David Silver UCL RL](https://www.davidsilver.uk/teaching/) : Man behind DeepMind's Go
5. [MIT Efficient Deep Learning](https://hanlab.mit.edu/courses/2023-fall-65940)
6. [MIT Introduction to Computational Thinking](https://computationalthinking.mit.edu/Fall20/) : Not exactly ML but very much relevant

### **Readings**

1. [Grokking Deep Learning ](https://edu.anarcho-copy.org/Algorithm/grokking-deep-learning.pdf)
2. [Understanding Deep Learning](https://udlbook.github.io/udlbook/) 
3. [Little Deep Learning Book](https://fleuret.org/public/lbdl.pdf)
4. [Introduction to RL Sutton & Barton](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

### **Blogs**

Blogs are the best way to learn about anything. You can find really good blogs on Medium and Towards Data Science. I will list here some amazing blogs that I have come across by amazing people.

1. [Lil's Log](https://lilianweng.github.io/): Researcher @ OpenAI, best blogs ever
2. [colah's blog](https://colah.github.io/): Co-founder Anthropic, again one of the best
3. [AI Summer](https://theaisummer.com/): You can learn about anything related to ML here 
4. [Distill Pub](https://distill.pub/): Exceptional Research blogs 
5. [NLP Course for You](https://lena-voita.github.io/nlp_course.html): Blog based course for NLP, pair it with cs224n
6. [Pincone Library](https://www.pinecone.io/library/): A good database of videos and blogs related to LLM and MultiModal Learning
7. [Spinning Up RL](https://spinningup.openai.com/en/latest/): Great RL resource by OpenAI
8. [e2eml](https://e2eml.school/blog.html): Compilation of large set of topics related to DL
9. [PyTorch Tutorials](https://pytorch.org/tutorials/)
10. [Hugging Face Docs](https://huggingface.co/docs/transformers/index) : Everything you need to Finetune your LLM

### **Github Repos**

1. [Practical DL](https://github.com/yandexdataschool/Practical_DL)
2. [Practical RL](https://github.com/yandexdataschool/Practical_RL)
3. Just google search "Awesome {}".format(Topic Name such as Deep Learning, Quant etc)

### **Competitions**

I have listed competitions as resources because they are the best way to learn. You can always participate in the Techfest and college competitions, but the place to really test it out are competitions @ **Kaggle and AI Crowd**. They host some of the best challenges, that require a dedicated team with experience. But you should always try, **"Shoot for the moon and land among the stars".**

**Conference Challenges:** One thing that not many people seem to know is that A* ML conferences such as **CVPR, and NeurIPS** conduct challenges that allow you to do research in a given framework and potentially lead to a paper. Again, they are very competitive and hard, but with the right guidance and hard work, you can do good and learn a lot about ML research.

Use this website to keep track of various ML competitions : [ML Contests](https://mlcontests.com/)

**Ps**: A person in **IIT BHU** secured **World Rank 2** in a **NeurIPS RL Challenge.** 

### **Computes**

I have talked about compute and that how they are expensive. But there are some resources that you can access for your use. 

1. **Google Colab**: Colab provides you with T4 GPU. Its a decent one and can be used for small training purposes. 
2. **Kaggle**
3. **SuperComputer**: You can get access of supercomputer in the institute which contains V100s as far I know. Allocation of GPUs depends on availability sometimes. 
4. **Microsoft Planetary Hub**: Not many know about this but MS provides you with an access to their planetary hub which have decent compute (better than T4 in Colab). You just need to fill a form and submit it takes a few days I guess. Allocation of GPU instance depends on availability and may take time. 
5. **Microsoft Azure**: You get some free credits with your institute ID with a limit of 3 months. You can use it for a project which involves some large training. 

### **Communities** 

You will be surprised to know that there are people around the world who are humble enough to help you and guide you. These people have formed open communities where they take sessions, do paper readings, and invite great researchers and do research projects and collaborations. You can literally just join their **discord server** and interact with so many amazing people. There are communities from **Cohere, EleutherAI** and many others. I have been a part of them and they are amazing. I have met some of our alumns there as well.

### **Inter IIT Tech Meet**

At last. **Inter IIT Tech Meet** is a prestigious competition where all IITs compete in various domains including ML. It's not a resource but if you make it to the team, you get to learn so much. It's like doing an internship. You get to do research for an accomplished company or lab such as **Adobe, Bosch, Devrev, and ISRO** and present in front of them. The PS is so relevant that you can, if did more work, make a research publication out of it. But the main motivation should here be doing quality research and winning it for your institution. Projects or Publications are by-products. It gives you a chance to go out there and be the best in your domain across all the IITs. I hope, if you have read this far then you will utilize these resources for your learning and gain the required skills to win **Gold @ Inter IIT**. **Good Luck!**

### **Regards**

Forgive me if I made any technical mistakes or for my grammar. The aim was to let others know about different domains and provide them with resources for learning and competing, that have been helpful to me or that I have come across. 

**With Regards,**\
**Dhruv Jain**\
**Electronics Engineering**
