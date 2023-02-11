import requests

request = "Milka and John are playing in the garden. Her little sister is playing too. " \
          "Milka is ready to start classes next week and it will be her first term in school. " \
          "In the morning, Milka gets up early to take a bath. " \
          "She puts on her school uniform and carries her school bag. " \
          "Her Mother gives her two thousand shillings for school fees and five hundred shillings for transport. " \
          "Then, she quickly goes to school. " \
          "Meanwhile, her big brother stays at home. He is still in his bed and sleeps. " \
          "Once she grows up and graduates school, Milka dreams to build a beautiful house for her and her family. " \
          "While she is at school, she is very active and participates in all the activities. " \
          "The teachers love her attitude. Milka listens carefully to her teacher. " \
          "Her classmates admire her too, because she is a kind girl. " \
          "At break time she tries to help other classmates with their practical exercies and homeworks."
print(requests.post('http://127.0.0.1:8000/prediction', json={'text': request}).content)
