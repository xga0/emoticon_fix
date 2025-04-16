from emoticon_fix import remove_emoticons

text = 'This message :D contains some (｡♥‿♥｡) emoticons that need to be removed!'
result = remove_emoticons(text)
print(result)  # Output: 'This message contains some emoticons that need to be removed!'



# from emoticon_fix import replace_emoticons

# # Default format: __EMO_{tag}__
# text = 'Happy customers :) are returning customers!'
# result = replace_emoticons(text)
# print(result)  # Output: 'Happy customers __EMO_Smile__ are returning customers!'

# # Custom format
# text = 'User feedback: Product was great :D but shipping was slow :('
# result = replace_emoticons(text, tag_format="<EMOTION type='{tag}'>")
# print(result)  # Output: 'User feedback: Product was great <EMOTION type='Laugh'> but shipping was slow <EMOTION type='Sad'>'