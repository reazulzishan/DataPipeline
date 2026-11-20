# The End-to-End Data Science Pipeline — YouTube tutorial + one Reel
Narrator: VCTK p270. Every number is from notebooks/pipeline_live.ipynb (Inside Airbnb NYC, snapshot 2026-09-14, CC BY 4.0).

[Y01]
Thirty five percent of these prices are missing. The rest are stored as text. And one column secretly contains the answer. This is real data, and today we take it all the way from a raw file to a live prediction.

[Y02]
Here is the whole pipeline. First, the question. Then ingest the data, validate it, clean it, and store it. Explore. Engineer features. Split the data. Build one pipeline. Then model, evaluate, explain, deploy, and monitor. Fifteen stages. Most tutorials skip half of them. We will not.

[Y03]
Stage one: the question. How much should an Airbnb listing in New York City charge per night? The metric: the average miss, in dollars. And one rule: the model must beat a simple baseline, on listings it has never seen.

[Y04]
Stage two: ingest. The raw file from Inside Airbnb has twenty nine thousand nine hundred and eighteen listings, ninety columns, sixteen megabytes. Rule number one: never edit the raw file. Every change happens on a copy, so you can always go back.

[Y05]
Look at the first rows. Three of the four prices are missing. The one that exists is text, with a dollar sign. Bathrooms are written as words. This is what real data looks like.

[Y06]
Stage three: profile and validate. Every bright cell here is a missing value. Thirteen columns are completely empty. And thirty five percent of the prices are missing.

[Y07]
Then we write rules that every row must pass. The price must exist and be above zero: ten thousand five hundred and twelve rows fail. The minimum stay must be under a year: twenty four fail. Location and IDs pass.

[Y08]
Now the important question: is the price missing at random? No. For short stays, eleven percent are missing. For stays of thirty nights or more, forty one percent. And eighty one percent of all listings require thirty nights, because of New York's short term rental law. Missing data has a reason. Find it.

[Y09]
Stage four: clean. One documented step at a time, with the row count after every step. First, drop the thirteen empty columns. Then keep only the twenty five columns we need.

[Y10]
Turn the price text into a number: the dollar sign and the comma come out. Then drop the rows with no price. We never invent the answer we are trying to predict. Nineteen thousand four hundred and six rows remain.

[Y11]
The bathroom column: one point five baths, shared with others, becomes the number one point five, plus a flag for sharing. Half bath becomes zero point five. The amenities are a list hidden inside a single text field. We count them, and pull out wifi, kitchen, air conditioning, washer and workspace.

[Y12]
Seven listings demand a minimum stay of more than a year. Impossible, so they go. The most expensive listing asks thirty one thousand five hundred dollars a night. Above two thousand dollars there are one hundred and ninety four listings: a different market. We keep them out, and we say so.

[Y13]
Duplicates: zero exact copies. But two thousand two hundred and eighty two rows look like twins: same host, same spot, same kind of room. They are real, separate apartments. We keep them, but remember them. They will matter at the split.

[Y14]
Stage five: store. The clean data is saved as a parquet file, named with the snapshot date. Sixteen megabytes became one point one. The price is a real number now. And the raw file is still untouched. A raw layer, and a clean layer.

[Y15]
Stage six: explore. Prices are skewed: most are cheap, and a few are very expensive. Taking the log of each price makes the shape almost symmetric. The skewness falls from two point five to zero point three. So we will predict the log of the price.

[Y16]
Manhattan is the most expensive borough. A private room there costs about two hundred and fifty dollars, two and a half times the same room in Brooklyn.

[Y17]
On the map, prices rise toward Midtown Manhattan. And here is a surprise: listings for under thirty nights cost a median of three hundred and seventy dollars. Thirty nights or more: one hundred and forty six.

[Y18]
Stage seven: feature engineering, turning knowledge into columns. The distance to Times Square, computed from latitude and longitude, has a correlation of minus zero point four two with the price. The closer a listing is, the higher its price. Add a flag for stays of thirty nights or more, and grouped property types: twenty seven features.

[Y19]
Stage eight: split, before anything learns. Remember the twins? With a random split, three hundred and forty two test listings have a near identical twin in training. The model would be tested on answers it has already seen. So we split by host. Twins in the test set: zero.

[Y20]
We also hold back one thousand five hundred and ten listings that appeared in twenty twenty six. The model never sees them, until we monitor it.

[Y21]
Stage nine: one pipeline. Missing values are filled with the median, and that median is learned from the training set only. Categories become one hot columns. Twenty seven input columns become one hundred and forty five model columns. Everything is stored in one object, so the same steps run in training and in production.

[Y22]
Stage ten: the leakage trap. Now we add one innocent looking column, the price quote per night. The model's R squared jumps to one. A perfect score. Too good to be true. It is the price itself, in disguise. So we remove it.

[Y23]
Stage eleven: model. First, a baseline. The median price for each borough and room type misses by one hundred and forty dollars. Ridge regression: eighty six. Light G B M: seventy three. Half the baseline's error.

[Y24]
And the twins? With a random split, Light G B M looks better: sixty six dollars. That number is a lie. Seventy three is the honest one.

[Y25]
Stage twelve: evaluate. Each dot is a test listing: the real price against the prediction. Sixty one percent are within twenty five percent of the real price. The biggest misses are in Manhattan, where prices vary the most: one hundred and five dollars on average.

[Y26]
A private room in Mill Basin: the real price is one hundred and thirteen dollars, the prediction one hundred and nine. An entire home in Midtown for six people: real, three hundred and ninety six. Predicted, four hundred and fifty five.

[Y27]
Stage thirteen: explain. Shuffle one column, and measure how much worse the model gets. The minimum stay matters most. That is the thirty night law, again. Then the distance to Times Square, shared bathrooms, and size.

[Y28]
Stage fourteen: deploy. The complete pipeline, with its cleaning, encoding and model, is saved as one file of four point six megabytes. Wrap it in a small web API. Send a listing, get a price: one hundred and nine dollars, in twenty milliseconds. The real price: one hundred and thirteen.

[Y29]
Stage fifteen: monitor. Data changes. For the listings that appeared in twenty twenty six, we compare every feature with the training data. The number of reviews shifted strongly. That is expected: new listings have few reviews. The minimum stay shifted too.

[Y30]
And the error on these new listings went up from seventy three to eighty five dollars. That is the signal to retrain. A model is never finished.

[Y31]
So that is the full pipeline. The question, ingest, validate, clean, store, explore, engineer features, split the data, build one pipeline, model, evaluate, explain, deploy, and monitor. The model is one stage out of fifteen. The rest is what makes it real.

[Y32]
Three rules to keep. Never edit raw data. Split before anything learns. And always compare against a baseline. The notebook and every result are on my GitHub.

# Reel (one standalone Reel, under 3 minutes)

[R01]
Thirty five percent of these prices are missing, and one column secretly contains the answer. Here is how real data becomes a working model, in fifteen stages.

[R02]
Load the raw file, and never edit it. It has twenty nine thousand listings and ninety columns. Then validate: thirteen empty columns, and ten thousand rows without a price.

[R03]
Missing is not random. Forty one percent of long stays hide their price, against eleven percent of short stays. Find the reason before you fix it.

[R04]
Clean, step by step. Prices written as text become numbers. One point five baths, shared with others, becomes one point five, plus a flag. Impossible stays and extreme prices are removed, and documented.

[R05]
Explore. Prices are skewed, so we predict the log of the price. And the closer to Times Square, the higher the price.

[R06]
Split before anything learns. A random split put three hundred and forty two near identical twins on both sides. Split by host: zero.

[R07]
The leakage trap. One column gave a perfect score. It was the price itself. Removed.

[R08]
Baseline: one hundred and forty dollars off. Light G B M: seventy three. Then deploy: one file, one web API, a price in twenty milliseconds.

[R09]
Then we monitor. On listings from twenty twenty six, the average miss went up to eighty five dollars. Time to retrain.

[R10]
Never edit raw data. Split before anything learns. Always compare with a baseline. The full tutorial is on YouTube. Which stage do you skip most often? Comment below.
