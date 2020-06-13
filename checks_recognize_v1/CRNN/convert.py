import collections 
import torch 



class strLabelConverter:
    """
    将文字和转解码成对应数字
    """


    def __init__(self, alphabet):
        self.alphabet = alphabet+'-'
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = int(i+1)
    def encode(self, text):
        length = []
        result = []
 
        for item in text:
            length.append(len(item))
            for char in item:
                #try:       
                index = self.dict.get(char, 1)
                result.append(index)
                # except Exception as e:
                #     print(e)
                #     result.append(0)
        text = result
              
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text, length, raw=False):
        if length.numel() == 1:
            length= length[0]
            if raw:
                return "".join(self.alphabet[i-1] for i in text)
            else:
                char_list = []
                for i in range(length):
                    if text[i]!=0 and (not (i > 0 and text[i-1] == text[i])):
                        char_list.append(self.alphabet[text[i]-1])
                return "".join(char_list)

        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(text[index:index+l], torch.IntTensor([l]), raw=raw)
                )
                index+=l
            return texts
    
    
