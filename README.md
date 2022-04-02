# Project Descriction

## crazytext

crazytext: An Easy To Use Text Cleaning Package For NLP Built In Python

#### Dependencies
```
pip install pandas
pip install numpy
pip install textblob
pip install sklearn
pip install lxml
pip install nltk
```

#### Installation

`pip install crazytext`




### Text Analysis Using crazytext

```python
sample_text = 'AI is the future of HUMAN KIND, & Trendiest Topic of Today. #ai #future @aiforfuture https://ai.com  (555) 555-1234  <p> Mobile Number </p> (555) 345-1234  <span>Pincode:</span> 224 '
```

**Let's Import Our Library**
```python
import crazytext as ct
```

* Quick Analysis

```python
doc = ct.Counter(text=sample_text)
doc.info()
>>
Length of String: 153
Number of URLs: 1
Number of Emails: 0
Number of Words: 25
Average Word Count: 6.12
Number of Stopwords: 4
Total Hashtags: 2
Total Mentions: 1
Total Length of Numeric Data: 7
Special Characters: 154
White Spaces: 28
Number of Vowels: 38
Number of Consonants: 143
Total Uppercase Words 3
Number of Phone Number Inside Text: 2
Observed Sentiment: (0.15, 'Positive')
```

* Step By Step Analysis

```python
doc.count_words()
>> 25

doc.count_stopwords()
>> 4

doc.count_phone_numbers()
>> 2

doc.count_uppercase_words()
>> 3

```
You Can Try Many More Methods Just Type `doc.count` and press `tab` to get all the available Counter Methods.

*Note : All The Methods For Counter Class Starts With `count_`*


### Text Extraction Using crazytext

```python
sample_text = 'AI is the future of HUMAN KIND, & Trendiest Topic of Today. #ai #future @aiforfuture www.ai.com (555) 555-1234  xyz@gmail.com <p> Mobile Number </p> (555) 345-1234  <span>Pincode:</span> 224 '
```

**Let's Import Our Library**
```python
import crazytext as ct
extractor = ct.Extractor(text=sample_text)
```

*Extracting Emails* 
```python
extractor.get_emails()
>>['xyz@gmail.com']
```

*Extracting Phone Numbers*
```python
extractor.get_phone_numbers()
['(555) 555-1234', '(555) 345-1234']
```

*Extracting UPPER CASE words*
```python
extractor.get_uppercase_words()
>>['AI', 'HUMAN', 'KIND,']
```


*Extracting Hashtags*
```python
extractor.get_hashtags()
>>['#ai', '#future']
```

*Extracting Mentions*
```python
extractor.get_mentions()
>>['@aiforfuture']
```

*Extracting HTML Tags*
```python
extractor.get_html_tags()
>>['<p>', '</p>', '<span>', '</span>']
```

Try Other Interesting Methods By Installing The Library Using `pip install crazytext`. 

*Note : All The Methods For Extractor Class Starts With `get_`*

### Text Cleaning Using crazytext

* There Are Two Ways To Clean The Text
1. Remove Text Completly.
2. Replace The Text With Its Saying

**1.  Remove Text Completly.**
```python
sample_text = '<h1>The Dark ó Knight</h1> a batman ó movie @batman ó #batman https://batman.com (555) 555-1234 ó 21 22 óó ó'
```
**Let's Import Our Library**
```python
import crazytext as ct
cleaner = ct.Cleaner(text=sample_text)
```

*Removing HTML Tags*
```python
cleaner.remove_html_tags_c()
>>' The Dark ó Knight a batman ó movie @batman ó #batman https://batman.com (555) 555-1234 ó 21 22 óó ó'
```

*Removing Phone Numbers*
```python
cleaner.remove_phone_numbers_c()
>> 'a batman ó movie @batman ó #batman https://batman.com  ó 21 22 óó ó'
```
**2. Replace The Text With Its Saying**
*Replacing HTML Tags*
```python
cleaner.remove_html_tags()
>>'HtmlTag The Dark ó Knight a batman ó movie @batman ó #batman https://batman.com (555) 555-1234 ó 21 22 óó ó'
```

*Replaxcing Phone Number*
```python
cleaner.remove_phone_numbers()
>> 'The Dark ó Knight</h1> a batman ó movie @batman ó #batman https://batman.com PhoneNumber ó 21 22 óó ó'
```

#### Quick Cleaning of A Document
To Clean A Doucment Quickly You Can Use `quickclean()` method inside `Cleaner` class.

*Quick Clean*
```python
import crazytext as ct
ct = Cleaner(text=sample_text)
ct.quickclean(remove_complete=True,make_base=False)
>>'the dark knight batman movie batman batman'
```
You Can Further Remove Duplicates Using The `remove_duplicate_words()` method. 

### Working With Dataframes Using crazytext
Let's Load `Hotel Reviews Dataframe` From My Github.
```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Abhayparashar31/NLPP_sentiment-analsis-on-hotel-review/main/Restaurant_Reviews.tsv',delimiter = "\t",quoting=3)
```

**Let's Import Our Library and Creat A Object For Our Class `Dataframe`**
```python
import crazytext as ct
dc = ct.Dataframe(df=df,col='Review')
```

Let's Find Our Dataframe Column Word Frequency Count Using crazytext

```python
dc.get_df_words_frequency_count()
>>
the             405
and             378
I               294
was             292
a               228
               ... 
Seat              1
dirty-            1
gross.            1
unbelievably      1
check.            1
Length: 2967, dtype: int64
```

**Cleaning The Dataframe Using One Line of Code With The Help of `pretty text`**

```python
df['cleaned_reviews'] = dc.clean(remove_complete=True,make_base='lemmatization')
df['cleaned_reviews']
>>
0                                        wow loved place
1                                         crust not good
2                                not tasty texture nasty
3      stopped late may bank holiday rick steve recom...
4                         the selection menu great price
                    ....
```

Next, Let's Convert This Cleaned Text Into Vectors For Further Processing
```python
vector = ct.Dataframe(df=df,col='cleaned_reviews')
vector.to_tfidf(max_features=3500)
>>
array([[0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.72888336, 0.6846379 , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       ...,
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ]])

```

#### FUTURE WORK
* More NLP Tasks To Be Added.
* Inbuilt Model Support To Be Added.



#### Uninstall
We Are Unhappy To See You Go, You Can Give Your Feedback By Putting A Comment On The Repo.

`pip uninstall crazytext`

#### Contributor
[Abhay Parashar](https://github.com/Abhayparashar31).
