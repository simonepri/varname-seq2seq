<h1 align="center">
  <b>varname-seq2seq</b>
</h1>
<p align="center">
  <!-- CI - TravisCI -->
  <a href="https://travis-ci.com/simonepri/varname-seq2seq">
    <img src="https://img.shields.io/travis/com/simonepri/varname-seq2seq/master.svg" alt="Build Status" />
  </a>
  <!-- License - MIT -->
  <a href="https://github.com/simonepri/varname-seq2seq/tree/master/license">
    <img src="https://img.shields.io/github/license/simonepri/varname-seq2seq.svg" alt="Project license" />
  </a>
</p>
<p align="center">
  📄Source code variable naming using a seq2seq architecture.
</p>


## Synopsis

varname-seq2seq is a source code sequence-to-sequence model that allows to train models to perform source code variable naming for virtually any programming language.

The image below shows an example of input for the model and the respective output produced.  
You can try a demo of this model for Java [using this Colab notebook][colab:demo-java].

<p align="center">
  <a href="#">
    <img height="350" src="https://user-images.githubusercontent.com/3505087/77019200-e6a7c200-6977-11ea-9b96-e51824ddcb62.png" alt="model" />
  </a>
</p>


## Variable Naming

By variable naming, we mean the task of suggesting the name of a particular variable (local variables and methods arguments) in a piece of code.
The suggested names should be ideally the ones that an experienced developer would choose in the particular context in which the variable is used.

For example, if the models receive the piece of code on the left, we may want him to suggest the correction on the right, in which we replaced `s` with `sum_of_squares`.

<table>
<tr>
<td>

```python
def score(X):
  s = 0.0
  for x in X:
    s += x * x
  return s
```

</td>
<td>

```python
def score(X):
  sum_of_squares = 0.0
  for x in X:
    sum_of_squares += x * x
  return sum_of_squares
```

</td>
</tr>
</table>


## Dataset generation

To train the model, we extract naming examples from a large corpus of several open-source projects in a given language.
A naming example is a piece of code in which we mask all the occurrences of a particular variable with a special `<mask>` token, and then we ask the model to predict the original variable name we masked.
When we generate naming examples, we can also obfuscate all the occurrences of surrounding variable names with the special `<varX>` to discourage the model from learning to name a variable relying on surrounding variable names.
In the following, we will be using the `obf` abbreviation to indicate that we used the obfuscation strategy just described.

Let us take the following piece of Java code as an example.
```java
public class Test { Test ( int a ) { int b = a ; } }
```
From this, we can extract two naming examples, one in which we mask all the occurrences of the variable `a`, and one in which we do the same but for the variable `b`.

```java
public class Test { Test ( int <mask> ) { int <var2> = <mask> ; } }
public class Test { Test ( int <var1> ) { int <mask> = <var1> ; } }
```

All these examples are divided into four splits.
We pick an arbitrary number of projects, and we use all the examples from these projects to create the `unseen` test set on which the model is tested. This set is made of projects from which the model has never seen any examples.
Then we use the remaining projects, and we randomly split all the examples extracted into the three balanced splits: train-dev-test.

### Pre-generated datasets

We distribute the pre-generated datasets showed in the table below.  
If you need more, you can generate new ones on your own by using [this Colab notebook][colab:dataset].

| Name | Language | Download |
|------|----------|----------|
| java-obf | Java | [![Download java-corpora-dataset-obfuscated.tgz](https://img.shields.io/github/downloads/simonepri/varname-seq2seq/latest/java-corpora-dataset-obfuscated.tgz.svg)][download:java-corpora-dataset-obfuscated.tgz] |
| java | Java | [![Download java-corpora-dataset.tgz](https://img.shields.io/github/downloads/simonepri/varname-seq2seq/latest/java-corpora-dataset.tgz.svg)][download:java-corpora-dataset.tgz] |


## Model training

The core idea of the model is to capture the syntactic usage context of a variable across a given fragment of code, and then to use this usage context to predict a natural name for a particular variable.
The intuition is that the usage context of a particular variable should contain enough information to describe how the variable is used, thus allowing us to derive an appropriate name.

This is achieved using two neural networks in an Encoder-Decoder architecture: one that condenses a sequence of tokens into an efficient vector representation that makes up the usage context, and another network that predicts a suitable name for the given usage context.

The image below shows a pictorial representation of the encoder-decoder model.  
`e` and `d` are two embedding layers, `z` is the usage context, and `f` is a linear layer.

<p align="center">
  <a href="#">
    <img height="250" src="https://user-images.githubusercontent.com/3505087/77015522-f1108e80-696c-11ea-837d-b5aa2328546c.png" alt="model" />
  </a>
</p>

### Pre-trained models

We distribute the pre-trained models showed in the table below.  
If you want to train the model on a different dataset, you can do so by using [this Colab notebook][colab:model].

| Name | Language | Download |
|------|----------|----------|
| java-obf | Java | [![Download java-lstm-1-256-256-dtf-lrs-obf.tgz](https://img.shields.io/github/downloads/simonepri/varname-seq2seq/latest/java-lstm-1-256-256-dtf-lrs-obf.tgz.svg)][download:java-lstm-1-256-256-dtf-lrs-obf.tgz] |
| java | Java | [![Download java-lstm-1-256-256-dtf-lrs.tgz](https://img.shields.io/github/downloads/simonepri/varname-seq2seq/latest/java-lstm-1-256-256-dtf-lrs.tgz.svg)][download:java-lstm-1-256-256-dtf-lrs.tgz] |


## Evaluation

To asses the effectiveness of the model, two primary metrics are considered: accuracy (ACC) and edit distance (EDIST).
Both metrics measure the ability of the model to recover the original names from the usage context of a particular variable, but they do so in a different manner.
The former measures exact target-prediction subword alignment, while the latter measures how many subword units need to be changed to transform the prediction in the target.

The following two figures show some simple examples of how the two metrics are computed.

<p align="center">
  <a href="#">
    <img height="100" src="https://user-images.githubusercontent.com/3505087/77015949-3b463f80-696e-11ea-86f5-2c72811e21c5.png" alt="model" />
    <img height="100" src="https://user-images.githubusercontent.com/3505087/77015962-426d4d80-696e-11ea-84b0-d36d936380ce.png" alt="model" />
  </a>
</p>


### Results

The following table reports the metrics for the different models-datasets we distribute.

| Model | Dataset | Test<br>ACC - EDIST | Unseen <br>ACC - EDIST | Test & Unseen <br>AVG |
|-------|---------|:-------------------:|:----------------------:|:---------------------:|
| java-obf | java-obf | 73.56% - 91.25% | **45.26%** - 80.92% | 72.75% |
| java | java | **73.54%** - 91.25% | 45.13% - **81.09%** | 72.75% |


## Authors

- **Simone Primarosa** - [simonepri][github:simonepri]

See also the list of [contributors][contributors] who participated in this project.


## License

This project is licensed under the MIT License - see the [license][license] file for details.


<!-- Links -->
[license]: https://github.com/simonepri/varname-seq2seq/tree/master/license
[contributors]: https://github.com/simonepri/varname-seq2seq/contributors

[src/bin]: https://github.com/simonepri/varname-seq2seq/tree/master/src/bin

[download:java-lstm-1-256-256-dtf-lrs-obf.tgz]: https://github.com/simonepri/varname-seq2seq/releases/latest/download/java-lstm-1-256-256-dtf-lrs-obf.tgz
[download:java-lstm-1-256-256-dtf-lrs.tgz]: https://github.com/simonepri/varname-seq2seq/releases/latest/download/java-lstm-1-256-256-dtf-lrs.tgz
[download:java-corpora-dataset.tgz]: https://github.com/simonepri/varname-seq2seq/releases/latest/download/java-corpora-dataset.tgz
[download:java-corpora-dataset-obfuscated.tgz]: https://github.com/simonepri/varname-seq2seq/releases/latest/download/java-corpora-dataset-obfuscated.tgz

[repo:Bukkit/Bukkit]: https://github.com/Bukkit/Bukkit
[repo:clojure/clojure]: https://github.com/clojure/clojure
[repo:apache/dubbo]: https://github.com/apache/dubbo
[repo:google/error-prone]: https://github.com/google/error-prone
[repo:grails/grails-core]: https://github.com/grails/grails-core
[repo:google/guice]: https://github.com/google/guice
[repo:hibernate/hibernate-orm]: https://github.com/hibernate/hibernate-orm
[repo:jhy/jsoup]: https://github.com/jhy/jsoup
[repo:junit-team/junit4]: https://github.com/junit-team/junit4
[repo:apache/kafka]: https://github.com/apache/kafka
[repo:libgdx/libgdx]: https://github.com/libgdx/libgdx
[repo:dropwizard/metrics]: https://github.com/dropwizard/metrics
[repo:square/okhttp]: https://github.com/square/okhttp
[repo:spring-projects/spring-framework]: https://github.com/spring-projects/spring-framework
[repo:apache/tomcat]: https://github.com/apache/tomcat
[repo:apache/cassandra]: https://github.com/apache/cassandra

[github:simonepri]: https://github.com/simonepri

[colab:demo-java]: https://colab.research.google.com/github/simonepri/varname-seq2seq/blob/master/examples/predict_java.ipynb
[colab:model]: https://colab.research.google.com/github/simonepri/varname-seq2seq/blob/master/examples/train.ipynb
[colab:dataset]: https://colab.research.google.com/github/simonepri/varname-seq2seq/blob/master/examples/dataset_generation.ipynb
