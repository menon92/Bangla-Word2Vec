# Bangla-Word2Vec
Bangla word2vec using skipgram approach. 

## Dataset
A toy bangla [dataset](./data/kothou_keu_nei_v1.2.txt) is used to training word vector. we create this data by taking all word from a famous bangla natok named `কোথাও কেউ নেই` . In early 90's it make a great hype among people. Some starting lines of this dataset,

`গেটের কাছে এসে মুনা ঘড়ি দেখতে চেষ্টা করল ডায়ালটা এত ছোট কিছুই দেখা গেল না আলোতেই দেখা যায় না আর এখন তো অন্ধকার রিকশা থেকে নেমেই একবার ঘড়ি দেখেছিল সাড়ে সাত গলির মোড় থেকে এ পর্যন্ত আসতে খুব বেশি হলে চার মিনিট লেগেছে কাজেই এখন বাজে সাতটা পঁয়ত্িশ এমন কিছু রাত হয়নি তবু মুনার অস্বস্তি লাগছে কালও ফিরতে রাত হয়েছে তার মামা শওকত সাহেব একটি কথাও বলেননি এমন ভাব করেছেন যেন মুনাকে দেখতেই পাননি আজও সে রকম করবেন ...`

## Training
Please use [traiing word2vec](./notebooks/word2vec_using_NCE_loss_v1_2.ipynb) or [colab](https://colab.research.google.com/drive/1I6dhrDAHU7j1nQIwfuaKElwKEGglt_dQ?usp=sharing) python notebook for training. Here is some sample code,

```python
x_test = np.array([word2id[w] for w in EVAL_WORDS])

start_time = time.time()
for step in range(1, NUM_STEPS+1):
    # ট্রেনিং এর জন্য ডাটা নেই 
    target_x, context_y = next_batch(BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW)
    
    with tf.GradientTape() as tape:
        # টার্গেট শব্দগুলোর বর্তমান এম্বেডিং বের করি
        emb = get_embedding(target_x)
        # এম্বিডিং এবং কন্টেক্সট থেকে লস হিসাব করি । 
        loss = nce_loss(emb, context_y)
    # loss এর সাপেক্ষে embedding, nce_weights, nce_biases ভেরিয়েবল গুলোর গ্রাডিয়েন্ট হিসাব করা 
    gradients = tape.gradient(loss, [embedding, nce_weights, nce_biases])
    # এই গ্রাডিয়েন্ট ধরে আমরা embedding, nce_weights, nce_biases ভেরিয়েবল গুলোর ভেলু আপডেট
    # করি 
    optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))
    
    # নিদিষ্ট স্টেপ পর পর আমরা লস দেখব 
    if step % DISPLAY_STEPS == 0 or step == 1:
        temp_emb = get_embedding(batch_x)
        loss = nce_loss(temp_emb, batch_y)
        print("Step: {} loss: {:.4f} time: {}".format(
            step, loss, time.time()-start_time)
        )
        start_time = time.time()
        
    # আমাদের সেট করা কিছু টেস্ট শব্দ দিয়ে আমরা টেস্ট করে দেখব আমাদের মডেল কেমন শিখতেছে
    if step % EVAL_STEPS == 0 or step == 1:
        print("Testing...")
        similarity = cosing_similarity(get_embedding(x_test)).numpy()
        for i in range(len(EVAL_WORDS)):
            top_k = 8  # আমরা কতগুলো নেয়ারেস্ট শব্দ দেখতে চাই সেটা সেট করে দিলাম 
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            log_str = "'{}' এর কাছের শব্দ গুলো: ".format(EVAL_WORDS[i])
            for k in range(top_k):
                log_str = "{} {},".format(log_str, id2word[nearest[k]])
            print(log_str)
  
# Training logs
Step: 1 loss: 115.20759582519531 time: 0.0004235943158467611
Testing...
'মুনা' এর কাছের শব্দ গুলো:  না, UNK, করে, হয়, তার, বলল, আছে, আমার,
'বকুল' এর কাছের শব্দ গুলো:  করে, তো, তুমি, হয়, কিছু, না, শুয়ে, UNK,
'বাকের' এর কাছের শব্দ গুলো:  সে, না, কিছু, UNK, তো, কথা, এই, মুনা,
Step: 10000 loss: 9.455401420593262 time: 2.2726904074350993
Step: 20000 loss: 7.710119724273682 time: 2.2709779103597003
Step: 30000 loss: 7.070896148681641 time: 2.2607158144315083
Step: 40000 loss: 6.505194664001465 time: 2.2212836742401123
Step: 50000 loss: 6.173004150390625 time: 2.214940134684245
Step: 60000 loss: 5.946339130401611 time: 2.2243704438209533
Step: 70000 loss: 5.727808952331543 time: 2.221015910307566
Step: 80000 loss: 5.547652244567871 time: 2.1990910172462463
Step: 90000 loss: 5.408536434173584 time: 2.1639158805211385
Step: 100000 loss: 5.279882431030273 time: 2.123226515452067
Step: 110000 loss: 5.184957027435303 time: 2.106153655052185
Step: 120000 loss: 5.0996479988098145 time: 2.108219035466512
Step: 130000 loss: 5.021121025085449 time: 2.108826458454132
Step: 140000 loss: 5.004368782043457 time: 2.113181738058726
Step: 150000 loss: 4.885924816131592 time: 2.1052868803342184
Step: 160000 loss: 4.8462138175964355 time: 2.1030974825223288
Step: 170000 loss: 4.791365623474121 time: 2.107759968439738
Step: 180000 loss: 4.754326820373535 time: 2.1050782958666483
Step: 190000 loss: 4.712499618530273 time: 2.107535886764526
Step: 200000 loss: 4.659974098205566 time: 2.097264516353607
Testing...
'মুনা' এর কাছের শব্দ গুলো:  বকুল, না, বাকের, করে, বলল, সে, কেন, UNK,
'বকুল' এর কাছের শব্দ গুলো:  মুনা, না, বলল, কেন, করে, UNK, বাকের, সে,
'বাকের' এর কাছের শব্দ গুলো:  মুনা, বকুল, না, করে, UNK, দিয়ে, সে, কথা,
Step: 210000 loss: 4.643957614898682 time: 2.0891512989997865
Step: 220000 loss: 4.634835720062256 time: 2.094070633252462
Step: 230000 loss: 4.586221694946289 time: 2.08938779036204
Step: 240000 loss: 4.5688323974609375 time: 2.095710357030233
Step: 250000 loss: 4.526012420654297 time: 2.115777929623922
Step: 260000 loss: 4.522487640380859 time: 2.1258484443028767
Step: 270000 loss: 4.494353771209717 time: 2.142400773366292
Step: 280000 loss: 4.480099678039551 time: 2.1417412400245666
Step: 290000 loss: 4.4421916007995605 time: 2.16306414604187
Step: 300000 loss: 4.403162479400635 time: 2.1549124240875246
Step: 310000 loss: 4.408456802368164 time: 2.133093810081482
Step: 320000 loss: 4.425971031188965 time: 2.1303428689638775
Step: 330000 loss: 4.391371726989746 time: 2.132253058751424
Step: 340000 loss: 4.366987228393555 time: 2.1308353583017987
Step: 350000 loss: 4.335179328918457 time: 2.116783610979716
Step: 360000 loss: 4.332767009735107 time: 2.122726575533549
Step: 370000 loss: 4.347087383270264 time: 2.1157851616541543
Step: 380000 loss: 4.347710609436035 time: 2.1155774076779683
Step: 390000 loss: 4.327152729034424 time: 2.1209163745244344
Step: 400000 loss: 4.28278923034668 time: 2.1125175317128497
Testing...
'মুনা' এর কাছের শব্দ গুলো:  বকুল, মামুন, না, বাকের, বলল, করে, সে, কেন,
'বকুল' এর কাছের শব্দ গুলো:  মুনা, কেন, না, বলল, আমি, বাবু, বাকের, করে,
'বাকের' এর কাছের শব্দ গুলো:  মুনা, বকুল, মামুন, UNK, দিয়ে, ভাই, না, করে, 
```

For pre-training embeddings check `embeddings/` directory

# Embeddings Vector visualization

Go to [https://projector.tensorflow.org/](https://projector.tensorflow.org/) and load your embeddings vector. It support only tsv format vector. Sample visualization,

![](./images/cosine_distance_between_baker_and_muna.png)


## Resoruces
- [Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf)
- [Skipgram with examples](https://www.tensorflow.org/tutorials/text/word2vec#skip-gram_and_negative_sampling)
- [Word2Vec Tensorflow 2x with low level api](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/word2vec.ipynb)
- [Word2Vec using Embedding Layer Tensorflow 2x](https://petamind.com/word2vec-with-tensorflow-2-0-a-simple-cbow-implementation/)
- [Word2Vec](https://www.tensorflow.org/tutorials/text/word2vec)
- [Word Embeddings](https://www.tensorflow.org/tutorials/text/word_embeddings)
- [skipgram function defination](https://keras.rstudio.com/reference/skipgrams.html)
