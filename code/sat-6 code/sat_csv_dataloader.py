# Define function to convert CSV data to image chips
def csv2img(csvData, outDir):
    data = pd.read_csv(csvData,header=None)
    data_rows, data_cols = data.shape
    out_dir = outDir
    for i in range(data_rows):
        row_i = data.iloc[i, 0:].to_numpy()
        im = np.reshape(row_i, (28,28,4))
        image_name = str(i)+'.png'
        output_path = os.path.join(out_dir,image_name)
        imsave(output_path, im)

# Subclass Dataset  ========================================================
class csv_data(Dataset):

    def __init__(self, df, data, transform=None):
        self.df = df
        self.directory = directory
        self.transform = transform

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 1]
        row_i = self.data.iloc[idx, 0:].to_numpy()
        image = np.reshape(row_i, (28,28,4))
        image = image.astype('uint8')
        image = torch.from_numpy(image)
        image = image.float()/255
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df)
