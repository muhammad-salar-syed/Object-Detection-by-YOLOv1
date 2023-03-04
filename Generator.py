

def ImageGenerator(image,label,batch_size):

    while True:
        start = 0
        end = batch_size

        while start < len(image):
            limit = min(end, len(image))
                       
            X = image[start:limit]
            Y = label[start:limit]

            yield (X,Y)   

            start += batch_size   
            end += batch_size