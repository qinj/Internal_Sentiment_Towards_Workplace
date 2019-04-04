def create_clean_glassdoor_reviews_csv():
    glassdoor_df = pd.read_csv('../data/google-amazon-facebook-employee-reviews/employee_reviews.csv', index_col=0)
    glassdoor_df.drop(['link'], axis=1, inplace=True)
    employee_titles_df = glassdoor_df["job-title"].str.split(" - ", n = 1, expand = True)
    employee_titles_df.columns = ['Current-Employee', 'role']
    employee_titles_df['is_current_employee'] = employee_titles_df['Current-Employee'].str.contains("Current Employee").astype(int)
    employee_titles_df.drop(['Current-Employee'], axis=1, inplace=True)
    glassdoor_df = pd.concat([glassdoor_df, employee_titles_df], axis=1)
    glassdoor_df['year'] = glassdoor_df['dates'].str.split(', ').str[1]
    for i, row in glassdoor_df.iterrows():
        if row['year'] == '0000' or pd.isnull(row['year']):
            glassdoor_df.drop([i], inplace=True)
    glassdoor_df['month'] = glassdoor_df['dates'].str.split(', ').str[0].str.split(' ').str[1]
    quarter_list = []
    for _, row in glassdoor_df.iterrows():
        if row['month'] == 'Jan' or row['month'] == 'Feb' or row['month'] == 'Mar':
            quarter_list.append(1)
        elif row['month'] == 'Apr' or row['month'] == 'May' or row['month'] == 'Jun':
            quarter_list.append(2)
        elif row['month'] == 'Jul' or row['month'] == 'Aug' or row['month'] == 'Sep':
            quarter_list.append(3)
        elif row['month'] == 'Oct' or row['month'] == 'Nov' or row['month'] == 'Dec':
            quarter_list.append(4)
        else:
            quarter_list.append(5)
    glassdoor_df['quarter'] = quarter_list
    glassdoor_df.columns = ['company', 'location', 'dates', 'job-title', 'summary', 'pros', 'cons',
                            'advice-to-mgmt', 'overall-ratings', 'work-balance-stars',
                            'culture-values-stars', 'career-opportunities-stars',
                            'comp-benefit-stars', 'senior-management-stars', 'helpful-count', 'role', 'is_current_employee', 'year', 'month', 'quarter']
    for i, row in glassdoor_df.iterrows():
        if row['overall-ratings'] == 'none':
            glassdoor_df.loc[i, 'overall-ratings'] = '-1'
        if row['work-balance-stars'] == 'none':
            glassdoor_df.loc[i, 'work-balance-stars'] = '-1'
        if row['culture-values-stars'] == 'none':
            glassdoor_df.loc[i, 'culture-values-stars'] = '-1'
        if row['career-opportunities-stars'] == 'none':
            glassdoor_df.loc[i, 'career-opportunities-stars'] = '-1'
        if row['comp-benefit-stars'] == 'none':
            glassdoor_df.loc[i, 'comp-benefit-stars'] = '-1'
        if row['senior-management-stars'] == 'none':
            glassdoor_df.loc[i, 'senior-management-stars'] = '-1'
    star_list = ['overall-ratings', 'work-balance-stars', 'culture-values-stars', 'career-opportunities-stars', 'comp-benefit-stars', 'senior-management-stars']
    for feature in star_list:
        glassdoor_df[feature] = glassdoor_df[feature].astype('float')
    for i, row in glassdoor_df.iterrows():
        if row['overall-ratings'] == -1:
            glassdoor_df.loc[i, 'overall-ratings'] = None
        if row['work-balance-stars'] == -1:
            glassdoor_df.loc[i, 'work-balance-stars'] = None
        if row['culture-values-stars'] == -1:
            glassdoor_df.loc[i, 'culture-values-stars'] = None
        if row['career-opportunities-stars'] == -1:
            glassdoor_df.loc[i, 'career-opportunities-stars'] = None
        if row['comp-benefit-stars'] == -1:
            glassdoor_df.loc[i, 'comp-benefit-stars'] = None
        if row['senior-management-stars'] == -1:
            glassdoor_df.loc[i, 'senior-management-stars'] = None
    glassdoor_df.to_csv('../data/clean_glassdoor_reviews.csv')
    return

def create_amazon_reviews_csv(glassdoor_filepath, amazon_earnings_filepath):
    glassdoor_df = pd.read_csv(glassdoor_filepath)
    q_earnings_df = pd.read_csv(amazon_earnings_filepath)
    amazon_reviews_df = glassdoor_df[glassdoor_df['company'] == 'amazon']
    earnings_list = []
    for _, row in amazon_reviews_df.iterrows():
        net_income = q_earnings_df['Quarterly Net Income (Billions)'].loc[(q_earnings_df['Quarter'] == row['quarter']) & (q_earnings_df['Year'] == row['year'])]
        if len(net_income.values) > 0:
            earnings_list.append(net_income.values[0])
        else:
            earnings_list.append(None)
    amazon_reviews_df['amazon_earnings_this_quarter'] = earnings_list
    overall_mean = np.nanmean(amazon_reviews_df['overall-ratings'])
    work_balance_mean = np.nanmean(amazon_reviews_df['work-balance-stars'])
    culture_mean = np.nanmean(amazon_reviews_df['culture-values-stars'])
    career_mean = np.nanmean(amazon_reviews_df['career-opportunities-stars'])
    benefit_mean = np.nanmean(amazon_reviews_df['comp-benefit-stars'])
    senior_mean = np.nanmean(amazon_reviews_df['senior-management-stars'])

    for i, row in amazon_reviews_df.iterrows():
        if row['overall-ratings'] != row['overall-ratings']:
            amazon_reviews_df.loc[i, 'overall-ratings'] = overall_mean

        if row['work-balance-stars'] != row['work-balance-stars']:
            amazon_reviews_df.loc[i, 'work-balance-stars'] = work_balance_mean

        if row['culture-values-stars'] != row['culture-values-stars']:
            amazon_reviews_df.loc[i, 'culture-values-stars'] = culture_mean

        if row['career-opportunities-stars'] != row['career-opportunities-stars']:
            amazon_reviews_df.loc[i, 'career-opportunities-stars'] = career_mean

        if row['comp-benefit-stars'] != row['comp-benefit-stars']:
            amazon_reviews_df.loc[i, 'comp-benefit-stars'] = benefit_mean

        if row['senior-management-stars'] != row['senior-management-stars']:
            amazon_reviews_df.loc[i, 'senior-management-stars'] = senior_mean
    amazon_reviews_df.to_csv('../data/clean_amazon_reviews.csv')
