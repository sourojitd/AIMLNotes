# 📚 My Introduction to Python Notes

---

## 🐍 Python Fundamentals

### **1. Variables**

**Detailed Explanation:**
[cite_start]Variables are used to store various types of data[cite: 371]. [cite_start]This includes single values (like integers, floats, strings, or booleans) and more complex data structures (like lists or dictionaries)[cite: 371]. [cite_start]You create a variable using the assignment operator (=)[cite: 372]. [cite_start]Python is dynamically typed, meaning it automatically assigns a data type to a variable based on the value you store in it[cite: 373]. [cite_start]You can check a variable's type at any time using the `type()` function[cite: 374].

> #### 🧠 Kid Card: The Labeled Box
>
> A **variable** is like a labeled storage box.
>
> * If you write `my_age = 10`, you're putting the number `10` into a box and sticking a label on it that says `my_age`.
> * If you write `my_name = "Alex"`, you're putting the word `"Alex"` into a box labeled `my_name`.
>
> The label lets you easily find what you stored later.

### **2. Operators**

**Detailed Explanation:**
Python supports different types of operators to perform actions:
* [cite_start]**Mathematical:** Used for arithmetic, such as `+` (addition), `-` (subtraction), `*` (multiplication), `/` (division), `%` (modulus), and `**` (exponentiation)[cite: 375].
* [cite_start]**Comparison:** Used to compare values, such as `==` (equal to), `!=` (not equal to), `>` (greater than), and `<` (less than)[cite: 376].
* [cite_start]**Membership:** Used to check if a value is in a sequence, such as `in` and `not in`[cite: 377].

> #### 🧠 Kid Card: Action Buttons
>
> Operators are like action buttons on a calculator or a game.
>
> * `+`, `-`, `*` are **math buttons** that add, subtract, and multiply.
> * `==`, `>` are **question buttons**. `5 > 2` asks, "Is 5 bigger than 2?" It returns `True`.
> * `in` is a **"search" button**. `"apple" in my_fruit_list` asks, "Is 'apple' inside the fruit list box?"

### **3. Data Structures (List, Tuple, Dictionary)**

**Detailed Explanation:**
[cite_start]Python has built-in data structures to organize collections of data[cite: 377, 378].

* [cite_start]**List:** A **mutable** (changeable), ordered collection of items[cite: 379]. [cite_start]They are defined with square brackets `[]`[cite: 379]. [cite_start]You can use methods like `.append()` to add an item [cite: 385] [cite_start]or `.pop()` to remove one[cite: 384]. [cite_start]You can also "slice" a list using indexing like `my_list[start:end]`, which includes the start index but excludes the end index[cite: 380, 381].
* [cite_start]**Tuple:** An **immutable** (unchangeable), ordered collection of items[cite: 386]. [cite_start]They are defined with parentheses `()`[cite: 386]. [cite_start]Once you create a tuple, you cannot change the elements inside it[cite: 387].
* [cite_start]**Dictionary:** A **mutable** (changeable), **unordered** collection of `key:value` pairs[cite: 388]. [cite_start]They are defined with curly brackets `{}`[cite: 388]. [cite_start]You access values using their unique key, not an index[cite: 389]. [cite_start]You can get a list of all keys with `.keys()` or all values with `.values()`[cite: 400].

> #### 🧠 Kid Card: Ways to Store Your Toys
>
> * **List (A LEGO Train):** It's a row of toy cars hooked together in order. You can *change* it: add a new car at the end (`.append()`), remove the last car (`.pop()`), or swap cars around. It's **mutable**.
>
> * **Tuple (A Super-Glued Train):** It's a LEGO train where all the cars are super-glued together. You can count them and look at them, but you *cannot* add, remove, or change any of them. It's **immutable**.
>
> * **Dictionary (A Locker Room):** It's a set of lockers. Each locker has a *key* (the locker number, like "Locker 10") and a *value* (your stuff inside, like "my gym bag"). You find your stuff by its key. The lockers aren't in any particular order.

### **4. Conditional Statements (if/elif/else)**

**Detailed Explanation:**
[cite_start]Conditional statements are used to make decisions in your code based on specific rules[cite: 402].
* [cite_start]The `if-else` construct is used for a single decision[cite: 402].
* [cite_start]The `if-elif-else` construct is used for more than one decision[cite: 403]. [cite_start]The `elif` (else if) statement is checked only if the first `if` condition (and any preceding `elif` conditions) are false[cite: 404].
* [cite_start]Python is a "tab-based language," so the indentation (the spaces) after the `if`, `elif`, or `else` is mandatory and defines the code block[cite: 405].

> #### 🧠 Kid Card: A "Choose Your Own Adventure"
>
> This is just a way to make choices.
>
> * **IF** it is raining:
>     * I will take an umbrella.
> * **ELIF** (else if) it is sunny:
>     * I will wear sunglasses.
> * **ELSE** (for any other situation):
>     * I will just wear a jacket.

### **5. Looping Statements (for/while)**

**Detailed Explanation:**
[cite_start]Loops are used to repeat a set of statements[cite: 406].
* [cite_start]**for loop:** This loop iterates through a sequence (like a list or a tuple), executing the code block for each element in the sequence[cite: 407].
* [cite_start]**while loop:** This loop requires a condition to be evaluated[cite: 408]. [cite_start]It will keep repeating the code block as long as the condition remains true[cite: 408].

> #### 🧠 Kid Card: Doing Chores
>
> * **for loop (Doing a specific list of chores):**
>     "For *every chore* on my list (list = [take out trash, wash dishes, walk dog]):
>     * Do that chore."
>     It repeats exactly 3 times, once for each item.
>
> * **while loop (Doing a chore until it's done):**
>     "While *the plate is still dirty* (condition = True):
>     * Keep scrubbing."
>     It repeats an unknown number of times and only stops when the plate is clean (condition = False).

### **6. Functions**

**Detailed Explanation:**
[cite_start]Functions are blocks of instructions that perform specific tasks[cite: 409]. [cite_start]They are great for code reusability and organization[cite: 409, 410]. [cite_start]You define a function using the `def` keyword[cite: 413].
* [cite_start]**Arguments:** Functions can take inputs, called arguments[cite: 416].
    * [cite_start]**Positional Arguments:** Arguments passed in order[cite: 417].
    * [cite_start]**Keyword Arguments:** Arguments passed with a name and a default value, and they must come after positional arguments[cite: 418, 419].
    * [cite_start]`*args` allows a function to accept any number of extra positional arguments[cite: 420].
    * [cite_start]`**kwargs` allows a function to accept any number of extra keyword arguments, which it stores as a dictionary[cite: 421].
* [cite_start]**Return:** A function can send a value back using the `return` statement, and it can even return multiple values at once[cite: 414, 415].

> #### 🧠 Kid Card: A Smoothie Recipe
>
> A **function** is like a reusable recipe.
>
> * `def make_smoothie(fruit, liquid):` This is the recipe name and the **ingredients** (arguments) it *needs*.
> * `*args` would be like letting someone add any number of "extra" toppings (like spinach, seeds, etc.).
> * `**kwargs` would be for special instructions (like `blender_speed="high"`).
> * The code inside is the *instructions* (add fruit, add liquid, blend).
> * `return my_smoothie` is the function *handing you* the finished smoothie.
>
> Now you can `make_smoothie("Banana", "Milk")` or `make_smoothie("Strawberry", "Orange Juice")` anytime you want, just by calling its name!

### **7. Regex (Regular Expressions)**

**Detailed Explanation:**
[cite_start]Regular Expressions (or Regex) are used for searching, matching, and manipulating text patterns[cite: 422]. [cite_start]You use Python's `re` module[cite: 422].
* [cite_start]`re.search()`: Finds a match anywhere in the text[cite: 423].
* [cite_start]`re.match()`: Only matches from the *start* of the text[cite: 428].
* [cite_start]`re.findall()`: Returns a list of all matches[cite: 429].
* [cite_start]`re.sub()`: Replaces matches with new text[cite: 430].
* [cite_start]**Common Patterns:** `\d` matches a digit, `\w` matches a word character, and `\s` matches a space[cite: 438].

> #### 🧠 Kid Card: A Super-Finder
>
> Regex is like the "Find" tool in a text editor, but on steroids.
>
> * Instead of searching for `"cat"`, you can search for a *pattern*.
> * If you want to find any 3-letter animal, you could search for `"cat|dog|pig"`.
> * If you want to find any phone number, you can tell it to find a *pattern* of 3 digits, then a dash, then 4 digits (like `\d\d\d-\d\d\d\d`). It's a way to find *shapes* of text, not just exact words.

---

## 🐼 NumPy and Pandas

### **1. NumPy (Numerical Python)**

**Detailed Explanation:**
[cite_start]NumPy (Numerical Python) is a library for numerical computing[cite: 441]. [cite_start]Its main object is the **ndarray** (n-dimensional array), which is a powerful grid for mathematical and statistical operations[cite: 441].
* [cite_start]**Key Difference:** Unlike Python lists, NumPy arrays can **only contain values of the same data type**[cite: 442].
* [cite_start]**Creating Arrays:** You can create arrays using `np.array()` (from a list) [cite: 444][cite_start], `np.arange(start, stop)` [cite: 445][cite_start], `np.linspace(start, stop, num)` (includes the `stop` value) [cite: 448][cite_start], and random generators like `np.random.rand()` (uniform distribution) [cite: 450] [cite_start]or `np.random.randn()` (normal distribution)[cite: 452].
* [cite_start]**Operations:** You can perform fast, element-wise arithmetic operations directly on arrays (e.g., `array1 * 2`)[cite: 462]. [cite_start]You can also change the array's dimensions with `.reshape()`[cite: 460].

> #### 🧠 Kid Card: The Egg Carton
>
> A normal Python **List** is like a junk drawer. It can hold anything: a sock, a battery, a key, a number.
>
> A **NumPy Array** is like an **egg carton**. It *only* holds eggs (one data type, like numbers).
>
> Because it's so organized, you can do things *really fast*. If you want to "add 2" to everything, you can just tell the carton "add 2," and it happens to all 12 eggs at once, instead of opening the junk drawer and adding 2 to every single item one by one.

### **2. Pandas Series**

**Detailed Explanation:**
[cite_start]A Pandas Series is a **one-dimensional labeled array** that can hold data of a single (homogeneous) type[cite: 467]. [cite_start]Think of it as a single column from an Excel spreadsheet[cite: 468]. [cite_start]The labels are called the **index**[cite: 468]. [cite_start]You can create a Series from a list or a NumPy array[cite: 469].

> #### 🧠 Kid Card: A Single Column
>
> A Pandas **Series** is like taking a *single column* from a spreadsheet.
>
> Imagine you just have the "Student Names" column, or just the "Test Scores" column. It's a list of items, but each item has a label (the index, which is like the row number).

### **3. Pandas DataFrame**

**Detailed Explanation:**
[cite_start]A DataFrame is a **two-dimensional tabular data structure**, just like an entire Excel spreadsheet[cite: 471]. [cite_start]It has labeled axes (rows and columns)[cite: 471]. [cite_start]You can create a DataFrame from many sources, like dictionaries, lists, or a collection of Series[cite: 472].

> #### 🧠 Kid Card: The Whole Spreadsheet
>
> A Pandas **DataFrame** is the *whole spreadsheet*. It's a collection of Series (columns) all put together in one table. It has rows and columns, and you can do powerful things to the whole table at once.

### **4. Accessing Data (.loc and .iloc)**

**Detailed Explanation:**
* [cite_start]**.loc[]:** Accesses elements using **label-based indexing**[cite: 475]. This means you use the name of the row or column. It is **inclusive** of the end label. [cite_start]For example, `df.loc[:100, 'Age']` would select rows from the beginning up to and *including* the label `100`[cite: 475, 476].
* [cite_start]**.iloc[]:** Accesses elements using **integer-based indexing** (the position number)[cite: 477]. This means you use the row or column number (starting from 0). It is **exclusive** of the end index. [cite_start]For example, `df.iloc[:100, 2:4]` would select rows from 0 to 99 (not including 100) and columns at position 2 and 3 (not including 4)[cite: 477, 478].

> #### 🧠 Kid Card: Two Ways to Get a Book
>
> Imagine a bookshelf.
>
> * **.loc (Label):** This is like asking for a book by its **Title**. "Please get me the book *labeled* 'Harry Potter'." `df.loc['Row_5', 'Age_Column']`
>
> * **.iloc (Integer):** This is like asking for a book by its **Position**. "Please get me the *3rd* book from the *5th* shelf." `df.iloc[4, 2]` (Remember, we start counting from 0!)

### **5. Key Pandas Functions**

**Detailed Explanation:**
Pandas has many useful functions for data manipulation:
* [cite_start]`pd.read_csv("file.csv")`: Loads data from a CSV file into a DataFrame[cite: 486].
* [cite_start]`df.head()` / `df.tail()`: Shows the first 5 or last 5 rows[cite: 488, 489].
* [cite_start]`df.shape`: Provides the dimensions (rows, columns)[cite: 490].
* [cite_start]`df.info()`: Gives details on data types and non-null (non-missing) values[cite: 491].
* [cite_start]`df.describe()`: Returns a statistical summary (mean, min, max, etc.) of numerical columns[cite: 492].
* [cite_start]`df.value_counts()`: Checks the count of all unique values in a column[cite: 495].
* [cite_start]`df.drop(labels, axis=1)`: Drops a column (axis=1) or row (axis=0)[cite: 497]. [cite_start]Using `inplace=True` modifies the original DataFrame instead of returning a new one[cite: 498].
* [cite_start]`df.groupby(['col_name'])`: Splits the data into groups to perform aggregate functions (like `.mean()`) on them[cite: 500].
* [cite_start]`pd.merge(df1, df2, on='col')`: Merges two DataFrames based on a common column, similar to a SQL join[cite: 502, 504].
* [cite_start]`df.isnull().sum()`: Checks for and counts all missing (null) values in each column[cite: 513].
* [cite_start]`df.fillna(value)`: Fills missing values with a specified value[cite: 517].

> #### 🧠 Kid Card: Spreadsheet Superpowers
>
> Pandas gives you a remote control for your spreadsheet:
> * `.head()`: "Show me the top 5 rows."
> * `.info()`: "Tell me about my data. What's in it? Is anything missing?"
> * `.describe()`: "Give me the quick math summary (average, min, max)."
> * `.value_counts()`: "In the 'Color' column, count how many 'Red', 'Blue', and 'Green' there are."
> * `.groupby('Color')`: "Make a separate group for all the 'Red' toys, a group for 'Blue', etc." Then you can ask for the `.mean()` price of each group.
> * `.fillna(0)`: "Find all the empty, missing cells and fill them with a 0."

---

## 📊 Exploratory Data Analysis (EDA)

### **1. What is EDA?**

**Detailed Explanation:**
[cite_start]Exploratory Data Analysis (EDA) is a critical first step in analyzing data[cite: 522]. [cite_start]Its main purpose is to understand the high-level structure, find patterns, and get an early understanding of the data[cite: 523]. [cite_start]The typical components of EDA include a Data Overview, Univariate Analysis (one variable), Bivariate/Multivariate Analysis (multiple variables), Missing Value Treatment, and Outlier Detection[cite: 524].

> #### 🧠 Kid Card: Detective Work
>
> **EDA** is like being a detective who just arrived at a crime scene (your dataset).
>
> You're not trying to solve the case yet. You're just *exploring*. You:
> 1.  **Count** everything (rows, columns).
> 2.  **Inspect** everything (what *kind* of data is this? numbers? words?).
> 3.  **Look for clues** (patterns, averages).
> 4.  **Find weird stuff** (missing evidence, things that look out of place).

### **2. Summary Statistics (Mean, Median, Mode)**

**Detailed Explanation:**
[cite_start]These are measures of central tendency, or where the "center" of your data is[cite: 526].
* [cite_start]**Mean:** The average of all values in a numerical attribute[cite: 537].
* [cite_start]**Median:** The middle value of a numerical attribute when all values are arranged in order[cite: 538].
* [cite_start]**Mode:** The most frequently occurring value (can be numerical or categorical)[cite: 539].

> #### 🧠 Kid Card: Allowance Averages
>
> You and 4 friends list your weekly allowance: `$5`, `$6`, `$7`, `$8`, `$100`.
>
> * **Mean (Average):** ($5 + 6 + 7 + 8 + 100) / 5 = **$25.20**. This number is high and doesn't feel right, because the $100 (an outlier) skewed it.
>
> * **Median (Middle):** You line them up in order: $5, $6, **$7**, $8, $100. The one in the *middle* is **$7**. This is a much better description of what a "typical" friend gets.
>
> * **Mode (Most Common):** If two friends got `$6`, then `$6` would be the mode.

### **3. Univariate & Bivariate Analysis**

**Detailed Explanation:**
* [cite_start]**Univariate Analysis:** Examines a **single variable** at a time to understand its distribution and central tendency[cite: 543]. [cite_start]The goal is to spot patterns in that one variable[cite: 544]. [cite_start]Common plots are histograms and box plots[cite: 544, 545].
* [cite_start]**Bivariate Analysis:** Analyzes how **two variables relate to one another**[cite: 550]. This is used to find associations and correlations. [cite_start]Common plots are scatterplots and heatmaps[cite: 551].

> #### 🧠 Kid Card: School Class Stats
>
> * **Univariate (One Variable):** You *only* look at the "Height" of your classmates. You make a histogram (a bar chart) to see how many kids are in the "4 feet" bucket, "5 feet" bucket, etc.
>
> * **Bivariate (Two Variables):** You look at "Height" *and* "Shoe Size" together. You make a scatterplot to see if taller kids also tend to have bigger feet. You're looking for a *relationship*.

### **4. Skewness**

**Detailed Explanation:**
[cite_start]Skewness measures how asymmetrical a variable's distribution is (how lopsided it is)[cite: 546].
* **Symmetric Distribution:** Mean = Median = Mode. [cite_start]The data is evenly distributed[cite: 549].
* **Positive (Right) Skew:** Mean > Median. [cite_start]The majority of data points are on the left, with a long "tail" to the right[cite: 547].
* **Negative (Left) Skew:** Mean < Median. [cite_start]The majority of data points are on the right, with a long "tail" to the left[cite: 548].

> #### 🧠 Kid Card: The Playground Slide
>
> Imagine the histogram bars are a playground slide.
>
> * **Symmetric:** A perfect, bell-shaped hill. (Mean = Median).
> * **Positive (Right) Skew:** The *stairs* are on the left and the long *slide* goes off to the right. Most data is bunched up on the left. (Mean > Median).
> * **Negative (Left) Skew:** The *stairs* are on the right and the long *slide* goes off to the left. Most data is bunched up on the right. (Mean < Median).

### **5. Missing Values**

**Detailed Explanation:**
[cite_start]Missing values (often shown as `NaN` or `None`) indicate the absence of data[cite: 563]. [cite_start]They are common and can significantly affect your analysis[cite: 564].
* [cite_start]**Treatment:** The method selected depends on the data and your goal[cite: 565].
    * [cite_start]**Dropping:** You can drop the rows or columns that have missing values[cite: 571].
    * **Imputation:** You can "fill in" the missing spots.
        * **Mean Imputation:** Replace with the average. [cite_start]Good for numerical data, but sensitive to outliers[cite: 567, 568].
        * **Median Imputation:** Replace with the middle value. [cite_start]Better for skewed data or data with outliers[cite: 569].
        * **Mode Imputation:** Replace with the most common value. [cite_start]Used for categorical variables[cite: 570].
* [cite_start]**Caution:** Imputing with a central value can distort the data's original distribution and variance, especially if many values are missing[cite: 572, 573].

> #### 🧠 Kid Card: The 100-Piece Puzzle
>
> You have a 100-piece puzzle, but 5 pieces are missing (`NaN`). What do you do?
>
> * **Dropping:** You throw the whole puzzle in the trash because it's incomplete. (This is often a bad idea!).
> * **Mean Imputation:** You look at all the *other* blue sky pieces, melt them down, and make a new "average blue" piece to plug the hole.
> * **Median Imputation:** A safer version of the "average blue" that isn't messed up by one weirdly dark blue piece.
> * **Mode Imputation:** If the missing piece is in the "grass" section, you just grab another "grass" piece and stick it in.

### **6. Outlier Detection and Treatment**

**Detailed Explanation:**
[cite_start]Outliers are data points that deviate significantly from the majority of other data points[cite: 575]. [cite_start]They can impact analysis and modeling[cite: 575].
* [cite_start]**Boxplots:** A boxplot visualizes the data's distribution using quartiles (Q1, Q2, Q3) and can identify potential outliers[cite: 562].
* [cite_start]**IQR Method:** A common rule for defining outliers is any point that is less than `Q1 - 1.5 * IQR` or greater than `Q3 + 1.5 * IQR`[cite: 583]. (IQR, or Interquartile Range, is Q3 - Q1).
* [cite_start]**Handling:** You can drop the outliers, replace them with null, or "cap" them by replacing them with the whisker bound values (`Q1 - 1.5*IQR` or `Q3 + 1.5*IQR`)[cite: 585, 588]. [cite_start]Domain knowledge is important to decide if it's a genuine data point or an error[cite: 587].

> #### 🧠 Kid Card: That One Rich Friend
>
> You ask 100 kids what their allowance is. 99 kids get between $5 and $10. One kid gets $500.
>
> That $500 is an **outlier**. It's way outside the "normal" range.
>
> A **Boxplot** is like a special fence. The main fence (the "box") contains the middle 50% of kids. The "whiskers" extend to catch the rest of the normal kids. The $500 kid is way outside the fence, so the boxplot flags him as an outlier (shown as a dot).

---

## ✍️ Analyzing Text Data

### **1. Text Preprocessing**

**Detailed Explanation:**
[cite_start]This is the process of preparing and refining raw text data[cite: 592]. [cite_start]It involves removing "noise" and standardizing the text to make it suitable for analysis[cite: 592, 593].
Key tasks include:
* [cite_start]**Lowercasing:** Converts all text to lowercase[cite: 595].
* [cite_start]**Removal of Special Characters:** Removes characters like `!`, `@`, `#`, etc.[cite: 596].
* [cite_start]**Stripping Extra White Spaces:** Removes spaces from the start and end of words[cite: 594].
* [cite_start]**Stopword Removal:** Removes common words (like "and", "the", "is") that appear frequently but don't add much contextual meaning[cite: 597, 599].
* [cite_start]**Stemming:** Reduces words to their root form (e.g., "running" becomes "run") to capture the core meaning[cite: 600].

> #### 🧠 Kid Card: Cleaning Up a Messy Sentence
>
> A computer can't understand a messy sentence. Preprocessing is "cleaning" the text.
>
> **Original:** "Wow! The BEST dogs are running!!"
>
> 1.  **Lowercase:** "wow! the best dogs are running!!"
> 2.  **Remove Punctuation:** "wow the best dogs are running"
> 3.  **Remove Stopwords:** "wow best dogs running"
> 4.  **Stemming:** "wow best dog run"
>
> Now the computer sees the main *idea* ("wow", "best", "dog", "run") without the extra noise.

### **2. Text Vectorization (Bag of Words)**

**Detailed Explanation:**
[cite_start]This is the process of representing text in a numerical format so a computer can understand it[cite: 609]. The **Bag of Words (BoW)** model is a common way to do this. [cite_start]It represents a document by counting the **frequency of each unique word**, completely ignoring grammar and word order[cite: 610, 611].

> #### 🧠 Kid Card: Counting Words in a Bag
>
> Computers don't read words, they read numbers. **Bag of Words** turns a sentence into a list of counts.
>
> First, you build a dictionary of *all* words in your documents: `["The", "cat", "dog", "ran", "sat"]`
>
> * `Sentence 1: "The cat sat."`
> * **Bag:** `{The: 1, cat: 1, dog: 0, ran: 0, sat: 1}`
>
> * `Sentence 2: "The dog ran."`
> * **Bag:** `{The: 1, cat: 0, dog: 1, ran: 1, sat: 0}`
>
> It's just a "bag" where you count the words. The order doesn't matter.

### **3. Sentiment Analysis**

**Detailed Explanation:**
[cite_start]Sentiment analysis is the process of analyzing text to determine the emotional tone it conveys—typically positive, negative, or neutral[cite: 612].
* [cite_start]**Lexicon-Based Approach:** This method uses a predefined dictionary (a "sentiment lexicon") where each word is already given a score (e.g., "happy" = +1, "bad" = -1)[cite: 614]. [cite_start]The overall sentiment of a text is calculated by summing or averaging the scores of its words[cite: 615]. [cite_start]**VADER** is a popular lexicon-based tool that is good for social media because it understands capital letters, punctuation, and modifiers (e.g., "very good")[cite: 616, 617].
* [cite_start]**Machine Learning-Based Approach:** This method trains a model on a dataset of texts that have already been labeled as positive or negative[cite: 618]. The model learns the patterns associated with each sentiment. [cite_start]This requires feature extraction, like using Bag of Words or word embeddings (like Skip-gram) to turn the text into numerical vectors first[cite: 619, 620].

> #### 🧠 Kid Card: The "Mood Reader"
>
> This is teaching a computer to "read the mood" of a sentence.
>
> * **Lexicon (Dictionary) Method:** The computer has a dictionary: "love" = `+2`, "great" = `+1`, "terrible" = `-1`, "hate" = `-2`.
>     * It reads your review: "I love this great movie!"
>     * It adds the scores: `(+2) + (+1) = +3`. That's a **positive** review!
>
> * **ML (Learning) Method:** You don't give it a dictionary. Instead, you *show* it 1,000 positive reviews and 1,000 negative reviews. It *learns* the patterns itself, just like a spam filter learns what spam "looks" like.
