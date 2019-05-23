def standardize_text(df, question_field):
    df[question_field] = df[question_field].str.replace(r"http\S+", "")
    df[question_field] = df[question_field].str.replace(r"http", "")
    df[question_field] = df[question_field].str.replace(r"@\S+", "")
    df[question_field] = df[question_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[question_field] = df[question_field].str.replace(r"@", "at")
    df[question_field] = df[question_field].str.lower()
    return df


def text_to_array(text):

    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeddings_index = np.load('/Users/j/Dropbox/Learn/kaggle_quora/src/features/embeddings_index.npy', allow_pickle=True)
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)


def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True:
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])
