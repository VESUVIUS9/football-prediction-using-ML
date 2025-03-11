Introduction 
Chapter 1: Project Research 
1.1 Inspiration about the project  
EA (Electronic Arts), the company behind the EA FC football game series, is known for 
using advanced simulations and algorithms to predict the outcomes of major football 
tournaments. In its game, EA FC, simulations are run to replicate real-world football 
scenarios, predicting outcomes based on team and player data. EA uses historical 
performance, player statistics, and team dynamics to simulate matches, providing an 
engaging and realistic experience for players. 
The company has successfully predicted the winners of the last four World Cups by 
running these simulations, which consider various factors such as team strength, player 
form, and match conditions. 
Inspired by EA’s approach, this project aims to replicate similar prediction models for real
world football match outcomes. By analysing historical data spanning decades, the project 
applies machine learning techniques to predict match results, focusing on key factors like 
offense, defense, and goalkeeper performance. The goal is to offer insights for better 
decision-making in football and sports analytics, similar to how EA FC uses simulations 
to enhance its game experience and predict tournament winners. 
1.2 Research on Project requirements 
We conducted comprehensive research to determine the most effective methods, 
algorithms, and tools for building a reliable football match prediction system. After 
evaluating various approaches, we identified Monte Carlo simulation as a key technique 
due to its ability to model uncertainty and simulate numerous match scenarios. By 
incorporating variables such as team performance metrics, historical match data, and FIFA 
rankings, Monte Carlo simulations allowed us to analyze a wide range of outcomes and 
predict match results effectively. 
9 
Additionally, we explored machine learning algorithms like logistic regression, Random 
Forest, and SGD SVM, assessing their strengths in handling classification tasks and 
identifying patterns within complex datasets. These algorithms were chosen for their 
ability to improve prediction accuracy and uncover the relationships between key factors 
influencing match outcomes. 
For the development environment, Python was selected for its robust ecosystem of 
libraries, including pandas, Scikit-learn, and Matplotlib, which facilitated efficient data 
analysis, machine learning implementation, and visualization. We also chose VS Code for 
coding and utilized Jupyter notebooks to structure and execute our code interactively, as it 
allows for seamless integration of data analysis, visualization, and machine learning tasks. 
To present the insights dynamically, we used Streamlit to build interactive dashboards, 
making it easier for users to explore and interpret the results. This combination of 
simulations, machine learning, and Python’s versatility provided a strong foundation for 
our project. 
Language: Python 
Libraries :  
Data Manipulation 
• numpy (np): For numerical computations and array manipulations. 
• pandas (pd): For handling tabular data and performing data analysis. 
Visualization 
• matplotlib.pyplot (plt): For creating static, interactive, and animated visualizations. 
• seaborn (sns): For high-level, aesthetically pleasing statistical data visualization. 
Machine Learning (scikit-learn) 
• sklearn (sl): open source ML library. 
• PCA: Dimensionality reduction technique to transform features. 
• GradientBoostingClassifier: Ensemble learning algorithm for classification tasks. 
• RandomForestClassifier: Ensemble-based classifier using decision trees. 
• LogisticRegression: Logistic regression model for binary classification. 
• SGDClassifier: Stochastic gradient descent-based classifier for large-scale learning. 
• accuracy_score: Metric to evaluate the accuracy of a model. 
10 
• confusion_matrix: To analyze model prediction results. 
• roc_curve: To compute and plot the ROC curve. 
• roc_auc_score: To calculate the Area Under the ROC Curve (AUC). 
• GridSearchCV: Hyperparameter tuning using exhaustive grid search. 
• RandomizedSearchCV: Hyperparameter tuning with randomized search. 
• train_test_split: To split data into training and testing sets. 
• Pipeline: To chain multiple preprocessing steps and a model into one pipeline. 
• PolynomialFeatures: To generate polynomial and interaction features for models. 
Utilities 
• Counter: To count occurrences of elements in a collection. 
• tqdm: To display progress bars for loops and processes. 
• tabulate: To format and display tabular data in a readable format. 
Additional Libraries 
• streamlit (st): For building interactive web-based data science applications. 
• datetime: To work with dates and times. 
• warnings: To manage and suppress warnings during code execution. 
User Interface:  Streamlit 
11 
Chapter 2: Data Collection & Data cleaning 
2.1 Data collection 
We used 3 datasets from Kaggle site and compiled them to form a single dataset. The 
dataset has a dimension of 23,921 x 25. It comprises of all the international matches played 
from 1993-2022 showing the home and away team, the goals they scored, the goals they 
conceded, in which city/continent it was played, the tournament type like was it a friendly 
or an official tournament. It also displayed their FIFA rank, FIFA points and various 
parameters like gk, midfield, defense and attack score which helped in predicting the match 
outcomes. You could refer to Table 1 to view the attributes and dimensions of the dataset 
used. 
Figure 1: Dimension of the dataset 
Table 1: The above two tables display the international matches that has been played from 
1993-2022. It also displays the different attributes like the match date, home and away team’s 
rank , their scores , total points and many more. 
2.2 Data cleaning 
We cleaned the data by handling missing values through imputation or removal, 
eliminating duplicates, and standardizing formats for consistency. Outliers were addressed 
by removal or capping, and irrelevant columns were dropped to reduce noise. Finally, 
numerical features were scaled to ensure compatibility with machine learning algorithms, 
resulting in a clean and reliable dataset for analysis. 
12 
2.3 Challenges faced 
Before acquiring the structured dataset used in this project, we attempted web scraping to 
collect match-related data, which presented several significant challenges. 
1. Scattered Data Sources: The data was not consolidated in a single location but was 
scattered across multiple websites, making it difficult to gather a complete and 
consistent dataset. Many links provided partial or incomplete information; for instance, 
one site might list the teams and scores but omit crucial details like match type or 
location, necessitating cross-referencing multiple sources to fill in the gaps. 
2. Inconsistent Website Structure: Many websites lacked clear or consistent HTML 
structures with proper div or class identifiers, making it harder to parse data accurately. 
This added complexity to the web scraping scripts, increasing the likelihood of errors. 
3. Time-Consuming Process: Scraping the required data was not only resource-intensive 
but also time-consuming. Writing scripts to extract data from various sources, cleaning 
and standardizing it, and then merging it into a usable format involved numerous steps 
and lines of code, many of which felt redundant. 
4. Technical Limitations: Using tools like Selenium for scraping presented additional 
hurdles. While Selenium is powerful for dynamic content, its bot-like behavior often 
led to websites blocking access, requiring us to repeatedly manage sessions and use 
proxies. Furthermore, Selenium's high CPU usage added strain to our systems, slowing 
down the process. BeautifulSoup, on the other hand, was slower and consumed 
significant memory, limiting its effectiveness for large-scale scraping tasks. 
5. Manual Interventions: Due to the scattered and incomplete nature of the data, we had 
to manually handle several aspects of the scraping process, such as making HTTP 
requests, managing sessions, and reformatting the data to make it usable. This added 
extra overhead to the already tedious process. 
13 
Chapter 3: Selection of algorithms 
3.1 Algorithms we used 
→Logistic regression 
we used Logistic Regression as one of the foundational models. Logistic regression is 
effective for predicting binary outcomes, in this case, whether a team will win a match 
or not. The model uses input features such as average rank, rank difference, point 
difference, home team goalkeeper score, away team goalkeeper score, and various 
performance scores (defense, offense, midfield) for both the home and away teams. 
These features directly impact the target variable, is_won, representing match 
outcomes. We fitted the logistic regression model on the training data using a Pipeline, 
which allowed us to manage the entire process of transforming features and training the 
model in an organized and systematic manner. 
Figure 2: All the input features that is responsible for 
influencing the outcome of a football match. 
→polynomial features 
To enhance the logistic regression model's ability to learn more complex relationships 
between features, we introduced Polynomial Features with a degree of 2. This 
transformation creates new features that capture interaction terms and higher-order 
terms, allowing the model to account for non-linear relationships in the data. This 
feature transformation significantly improved the model's predictive power, especially 
when dealing with patterns that were not linear. 
→pipeline 
The Pipeline we used in the workflow combined both polynomial feature generation 
and logistic regression into a seamless process, ensuring all steps happened in the 
correct order. This prevented the model from being trained on the entire dataset, which 
14 
would otherwise lead to data leakage. It also ensured that only the training set was used 
during the fitting phase, leading to a more reliable model evaluation. 
→logistic regression with hyperparameter tunning 
We also performed Hyperparameter Tuning to find the optimal settings for the logistic 
regression model. Hyperparameters such as the regularization strength (C) and the 
solver type (either loglinear or saga) were adjusted to improve model performance. The 
loglinear solver is well-suited for small datasets and binary classification tasks, while 
saga is better for handling large datasets and more complex problems. By tuning these 
parameters, we ensured the model avoided underfitting (too simple) or overfitting (too 
complex), leading to better generalization on unseen data. 
Figure 3: we adjusted the regularization strength and the algorithm solver to improve 
the model’s performance. 
→random forest classifier 
The Random Forest Classifier is an ensemble learning technique that builds multiple 
decision trees during training and outputs the mode (most common) class of the 
individual trees. Each tree in the forest is trained on a random subset of the data with 
bootstrapping, and features are randomly selected for each split. This randomness helps 
reduce the variance of the model, making it more robust and less prone to overfitting 
compared to a single decision tree. We used the Random Forest model because it can 
handle both classification and regression tasks efficiently and works well with complex, 
high-dimensional datasets. It’s particularly useful for datasets with multiple features 
and interactions between them. Random forests also provide feature importance scores, 
which help us identify the most influential factors in predicting the outcome of a 
football match. 
→ Stochastic Gradient Descent – support vector machine 
The Stochastic Gradient Descent (SGD) combined with Support Vector Machine 
(SVM) is a powerful machine learning algorithm used for classification tasks. SVM 
aims to find the hyperplane that best separates the classes in the feature space, with the 
15 
largest margin between them. SGD is an optimization method used to update the 
model's parameters iteratively, making it highly efficient for large datasets. We chose 
the SGD SVM because it is suitable for high-dimensional data, which is often the case 
when dealing with multiple features, such as team performance scores and rankings. 
SVMs are known for their ability to find complex boundaries in data, making them 
effective for identifying patterns in match outcomes that may not be linearly separable. 
The model can be further optimized with the choice of kernel functions (e.g., linear, 
polynomial) and hyperparameters. 
Figure 4: use of PCA for dimension reduction (5 components/features)   
→ Gradient Boosting Classifier 
The Gradient Boosting Classifier (GBC) is another ensemble learning method, but 
unlike Random Forest, it builds trees sequentially, where each tree corrects the errors 
of the previous one. It uses gradient descent to minimize the loss function, iteratively 
fitting new trees to the residual errors of the current model. The Gradient Boosting 
model is highly effective for complex datasets where other models may struggle to 
capture non-linear relationships. We used Gradient Boosting because it often produces 
highly accurate results by combining weak learners into a strong model. This technique 
is particularly useful in predicting match outcomes where various factors interact in 
non-linear ways. GBC can also be fine-tuned for performance by adjusting parameters 
such as the number of estimators, learning rate, and maximum depth of trees. It is 
known for its high predictive accuracy and can handle both classification and regression 
problems efficiently. 
We split the dataset into training and test sets using a Train-Test Split, where 40% of the data 
was reserved for testing. This division allowed us to evaluate the model’s performance on 
unseen data, ensuring its ability to generalize beyond the training set. 
Finally, we used the ROC Curve and AUC Score to assess the model's performance. The ROC 
curve plots the true positive rate against the false positive rate at various threshold settings, 
while the AUC score provides a quantitative measure of the model's ability to discriminate 
16 
between positive and negative outcomes. A score of 0.5 suggests no discrimination, a score 
between 0.7 and 0.8 is considered acceptable, and a score between 0.8 and 0.9 is excellent. 
These metrics were critical in understanding the effectiveness of the logistic regression model 
in predicting match outcomes. 
3.2 Challenges faced 
• The dataset contained many features, leading to potential overfitting in models like 
Random Forest and Gradient Boosting. 
• Reducing dimensionality without losing important information proved difficult. 
• Models like Random Forest and Gradient Boosting were prone to overfitting, 
particularly with a large number of trees or deep trees. 
• Achieving the right balance between underfitting and overfitting was challenging. 
• Hyperparameter tuning for algorithms like Gradient Boosting and Logistic Regression 
was computationally expensive and time-consuming. 
• Training models like Random Forest and Gradient Boosting took a long time, especially 
with larger datasets. 
• The need for faster training times led to challenges in optimization. 
• The dataset was imbalanced, leading to biased predictions in models. 
• Handling class imbalance effectively while maintaining performance was a significant 
challenge. 
• Models like Random Forest and Gradient Boosting were difficult to interpret and 
explain, especially for feature importance and prediction results. 
• Determining the right evaluation metrics was crucial, and balancing between accuracy 
and model robustness was challenging. 
• Ensuring models could generalize well across different datasets and thresholds was 
difficult. 
17 
Chapter 4: Simulation 
We performed a Monte Carlo simulation to predict the outcomes of a football tournament, 
simulating each stage from the Round of 16 to the final. The simulation runs for 1000 iterations, 
generating different potential outcomes for each match based on a series of factors influencing 
the match results. 
The tournament involves 32 teams, with each stage progressively narrowing the field. The 
simulation begins with the Round of 16, where 16 pairs of teams compete against each other. 
In each match, the winner is determined based on a probability model (best_model), which 
uses several input features such as rank difference, point difference, and team-specific scores 
like goalkeeper score, defense score, offense score, and midfield score. These features help 
predict the likelihood of a home team winning against an away team. A random binomial 
distribution is used to simulate the actual outcome based on the predicted probability. 
Figure 5: we ran the simulation for 1000 times and also we can see all the 32 
participating teams  
As teams win and advance through the tournament, the Quarterfinal, Semifinal, and Final 
stages follow. For each of these stages, the winning teams from the previous round play against 
each other, with the process repeating similarly. Each match's outcome is simulated using the 
same model, ensuring consistency throughout the tournament simulation. 
The results from each stage, including the winners and their corresponding probabilities, are 
collected and stored. This data is captured in DataFrames, which are created for each stage of 
the tournament (Round of 16, Quarterfinal, Semifinal, and Final). The data includes columns 
for the stage, the winning team, and the probability of that team winning, providing a detailed 
record of each simulated match outcome. 
After completing all 1000 simulations, the results are consolidated into four final DataFrames, 
representing the outcomes at each stage of the tournament. These results can be analyzed to 
determine which teams are most likely to progress through each round and potentially win the 
entire tournament. 
18 
The key challenge of this simulation lies in the accuracy of the input data. The predictions are 
highly dependent on the quality of the team rankings and performance metrics used to train the 
model. If the data is not accurate or up-to-date, the simulation results may not be reliable. 
Additionally, the performance of the model itself can affect the outcomes; if the model is 
overfitted or underfitted, it may lead to inaccurate predictions. 
19 
Chapter 5: UI 
The Tournament Simulation UI enables users to simulate a football tournament by uploading 
a CSV file containing team data. The file should have specific columns like 
'home_team_fifa_rank', 'away_team_fifa_rank', 'home_team', and 'away_team'. The app 
checks for these required columns before proceeding. Once the dataset is validated, users can 
select 32 teams from the available teams in the dataset to participate in the tournament. The 
app then calculates each team's win probability based on their FIFA rankings and normalizes 
these probabilities for home and away games. 
After the teams are selected and win probabilities are calculated, the app runs the tournament 
simulation. The simulation involves grouping the teams into four groups, followed by knockout 
stages (Round of 16, Quarter-finals, Semi-finals, and Final). For each match, the winner is 
determined by a random process weighted by the calculated win probabilities. After running 
the specified number of simulations, the results are displayed, including the team that won the 
most simulations. The app also provides a summary of the selected teams with updated win 
probabilities. 
Match Analysis UI 
The Match Analysis UI allows users to analyze the performance statistics of any team based 
on historical match data. The data is loaded from a CSV file containing information about 
various football matches, including home team results, away team results, and performance 
metrics like midfield, defense, and offense scores. Users can select a team from a dropdown 
menu, and the app will calculate and display several statistics related to the team's performance, 
such as the total number of games played, wins, losses, draws, home/away performance, FIFA 
ranking, and specific scores for midfield, defense, and offense. 
In addition to the statistics, the app visualizes the team's performance through a pie chart that 
shows the percentage distribution of wins, losses, and draws. This feature helps users gain 
insights into a team's overall performance and their success rates in different types of matches. 
20 
Main Streamlit App Structure 
The main function of the app offers two main options through a sidebar: Tournament 
Simulation and Match Analysis. Depending on the user's choice, the app either directs them 
to the tournament simulation feature or the match analysis feature. This streamlined navigation 
ensures that users can easily access the functionality they need without being overwhelmed by 
multiple options on the main screen. 
Fig 6: Displaying the UI  
21 
22 
 
Chapter 6: Results and Discussions 
As we made our analysis, we came to know that the home team has the most chances of winning 
when the match is being played at their home ground with a win percentage of 65.79%, the 
away team won only 30.47 % of the time. 
 
We came to know that Spain is one of the best teams as they have the highest midfield, defense 
and gk scores. Argentina with the offense score of 88.25 is the best attacking team. 
We came to know that Brazil, Spain, Argentina, Portugal, France are the teams with the most 
home win rates. Teams like Brazil, Spain, and Argentina demonstrate strong overall win 
percentages, with Brazil having a notable home win percentage of 77.68%, reflecting a 
significant home advantage. Meanwhile, the data also reveals how teams like Tunisia and Saudi 
Arabia perform under different conditions, showing variations between home and away 
matches. This information is valuable for analyzing team strengths, understanding home versus 
away dynamics, and making predictions. 
Fig 7: Displaying the win %age of home and away teams  
       
Table 2: Displaying top 20 teams with their midfield, defense, gk and attacking score 
respectively. 
Table 3: Displaying top 20 teams with the number of wins, draw and loss  
The performance comparison of various machine learning models shows some nuanced 
differences in their ability to classify accurately, as measured by accuracy and AUC (Area 
Under the ROC Curve). Gradient Boosting stands out slightly, achieving the highest accuracy 
at 68.62% and an AUC of 0.75, indicating it performs marginally better in correctly classifying 
data points. The Logistic Regression models, both standard and tuned, follow closely with 
accuracies of 68.19% and 68.06%, respectively, and both achieve an AUC of 0.75. This 
suggests that while these models are relatively straightforward, they provide robust 
classification capabilities comparable to Gradient Boosting. 
The Random Forest model, however, lags a bit behind with an accuracy of 65.16% and an AUC 
of 0.70, indicating it might be less effective at distinguishing between classes in this dataset. 
The SGD SVM with PCA transformation achieves a slightly better accuracy than Random 
Forest at 67.15% and an AUC of 0.72, showing that while it benefits from dimensionality 
reduction, its performance still falls short of the Logistic Regression and Gradient Boosting 
models. 
Overall, the differences in performance are minor, with most models reaching similar accuracy 
and AUC values, particularly the top three models (Gradient Boosting and the two Logistic 
Regression models). This similarity in results might imply that the dataset or feature set has a 
certain degree of complexity that these models are approaching similarly. To achieve a more 
substantial improvement, further feature engineering, trying other model architectures, or 
exploring ensemble techniques may be necessary to capture additional patterns within the data. 
23 
Fig 8: displaying the AUC of all the different models that we trained 
Fig 9: displaying the important features that 
help in making contribution to the model’s 
prediction. 
Higher importance means the feature has a 
stronger influence on the model's decision
making process. 
The feature importance values indicate which factors most significantly influence the model's 
predictions. The top three features are rank_difference (0.4238), point_difference (0.1310), and 
average_rank (0.0642), making them the primary drivers of the model’s performance. These 
features reflect the competitive gap between teams, with rank_difference being the most 
crucial. Secondary features, such as offense scores, midfield scores, and goalkeeper scores for 
both home and away teams, have lower but meaningful importance. This analysis highlights 
that team rankings and point differences are far more impactful than individual player 
performance metrics in determining match outcomes. 
The bar charts illustrate the progression of teams through the stages of the 2022 World Cup, 
highlighting their performance consistency across rounds. In the "Road to Quarterfinals" chart 
(top left), USA and England have the highest counts, indicating a strong presence up to this 
stage, followed by teams like Belgium and Netherlands. Moving to the "Quarterfinal World 
Cup 2022" chart (top right), Netherlands leads with the highest count, closely followed by 
24 
25 
 
Mexico, Germany, Argentina, and Brazil, showing these teams' strength in reaching the 
Quarterfinals. In the Semifinals (bottom left), only four teams—Germany, Argentina, Brazil, 
and Portugal—remain, all with similar counts, demonstrating their competitive edge. Finally, 
in the "Winner World Cup 2022" chart (bottom right), only Brazil and Argentina appear with 
equal counts, suggesting they were the leading contenders for the championship. This 
progression analysis underscores the consistent performance of certain teams across stages, 
particularly Argentina and Brazil. 
 
 
 
Fig 10: output of the simulation showing the results of various stages of the tournament i.e. Road 
to QF, QF, SEMI and Final  
Conclusion 
The analysis of the football match data reveals several key insights. Brazil stands out 
as one of the strongest teams, ranking highly in both offensive and defensive metrics, 
with significant home advantage. Teams like Spain, France, and Argentina also 
demonstrate strong performances, particularly in defense and overall win percentages. 
Spain excels in goalkeeper scores, reflecting their solid defensive foundation. The data 
further indicates that home teams generally perform better than away teams, with Brazil 
and Spain showing high home win percentages. Additionally, Argentina and Qatar excel 
in offensive capabilities, while teams like Serbia show remarkable defensive efficiency, 
with a low "conceded goals per goalkeeper score" ratio. The win streaks of top teams 
like Spain and Brazil reflect their consistency and resilience over time. Overall, the 
analysis emphasizes the importance of both offensive strength and defensive stability, 
alongside the significant impact of playing at home in football outcomes. 
The simulation results for the 2022 FIFA World Cup indicate that Argentina emerged 
as the winner, showcasing the strongest performance throughout the tournament. Brazil 
was a close contender, consistently advancing through the stages. In the Road to 
Quarterfinals, teams like USA, England, and Netherlands performed notably well, 
while Senegal, Ecuador, and Portugal led in the Quarterfinals, with Brazil, Argentina, 
Germany, Mexico, and Netherlands also progressing strongly. In the Semifinals, 
Germany, Brazil, Argentina, and Portugal were the top contenders, with Germany 
showing a particularly strong presence. Ultimately, Brazil and Argentina stood out as 
the most dominant teams, with Argentina clinching the title in the final simulation, 
reflecting their overall superiority in the competition. 
