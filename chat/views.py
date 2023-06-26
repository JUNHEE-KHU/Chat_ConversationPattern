from django.shortcuts import render, redirect
from chat.models import Room, Message, User, UserFormalInformal
from django.http import HttpResponse, JsonResponse

from django.contrib import messages

import transformers
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from chat.utils import clean

from konlpy.tag import Okt


def home(request):
    return render(request, 'home.html')

def room(request, room):
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    return render(request, 'room.html', {
        'username': username,
        'room': room,
        'room_details': room_details
    })

def checkview(request):
    room = request.POST['room_name']
    username = request.POST['username']

    if Room.objects.filter(name=room).exists():
        return redirect('/'+room+'/?username='+username)
    else:
        new_room = Room.objects.create(name=room)
        new_room.save()
        return redirect('/'+room+'/?username='+username)

def send(request):

    BASE_DIR = Path(__file__).resolve().parent
    # model_token = os.getenv('MODEL_TOKEN')

    latest_model_path = str(BASE_DIR) + '/saved_model/checkpoint-17754'
    device = 'cpu'

    # pipeline = transformers.pipeline(
    #     "text-classification", model=model, tokenizer=tokenizer)

    class FormalClassifier(object):
        def __init__(self):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                latest_model_path).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

        def predict(self, text: str):
            text = clean(text)
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
            input_ids = inputs["input_ids"].to(device)
            token_type_ids = inputs["token_type_ids"].to(device)
            attentsion_mask = inputs["attention_mask"].to(device)

            model_inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attentsion_mask,
            }
            return torch.softmax(self.model(**model_inputs).logits, dim=-1)

        def is_formal(self, text):
            if self.predict(text)[0][1] > self.predict(text)[0][0]:
                return True
            else:
                return False

        def formal_percentage(self, text):
            return round(float(self.predict(text)[0][1]), 2)

        def formal_informal_which(self, text):
            result = self.formal_percentage(text)
            # print(result)
            if result > 0.5:
                formal_informal_which = 'formal'
            if result < 0.5:
                formal_informal_which = 'informal'
            return formal_informal_which
            
        def formal_informal_percent(self, text):
            result = self.formal_percentage(text)
            # print(result)
            if result > 0.5: # formal
                formal_informal_percent = result * 100
            if result < 0.5: # informal
                formal_informal_percent = result * 100 # 존댓말일 확률
                # formal_informal_percent = (1 - result) * 100
            return formal_informal_percent

    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']

    classifier = FormalClassifier()
    formal_informal_which = classifier.formal_informal_which(message)
    formal_informal_percent = classifier.formal_informal_percent(message)


    msg = ''

    # 존댓말/반말 count start
    if UserFormalInformal.objects.filter(room = room_id, user = username).exists():
        new_user_formal_informal = UserFormalInformal.objects.get(room = room_id, user = username)
        if formal_informal_which == 'formal':
            new_user_formal_informal.formal_count = new_user_formal_informal.formal_count + 1
            new_user_formal_informal.save()
        if formal_informal_which == 'informal':
            new_user_formal_informal.informal_count = new_user_formal_informal.informal_count + 1
            new_user_formal_informal.save()
        
        new_user_formal_informal.formal_percent_avg = (float(new_user_formal_informal.formal_count) / float((new_user_formal_informal.formal_count+new_user_formal_informal.informal_count)))*100
        new_user_formal_informal.save()
    else:
        if formal_informal_which == 'formal':
            new_user_formal_informal = UserFormalInformal.objects.create(room=room_id, user=username, formal_count=1, informal_count=0, formal_percent_avg=100)
        if formal_informal_which == 'informal':
            new_user_formal_informal = UserFormalInformal.objects.create(room=room_id, user=username, formal_count=0, informal_count=1, formal_percent_avg=0)

    new_user_formal_informal.save()

    # 존댓말/반말 end


    # 유사단어 start
    okt = Okt()
    nouns = okt.nouns(message)

    father = ['아빠', '아버지', '아부지', '아부이', '아방', '대디']
    mother = ['엄마', '어머니', '어무니', '어무이', '어망', '마미']
    sister = ['언니', '언닝', '온니', '온닝', '언뉘', '온뉘']

    all = father + mother + sister

    for noun in nouns:
        if noun in all:
            if User.objects.filter(room = room_id, user = username, voca = noun).exists():
                user_count = User.objects.get(room = room_id, user = username, voca = noun)
                user_count.count_var = user_count.count_var + 1
                user_count.save()
            else:
                new_user = User.objects.create(room = room_id, user = username, voca = noun, count_var = 1)
                new_user.save()
    # 유사단어 end

    new_message = Message.objects.create(value=message, user=username, room=room_id, formal_informal_which=formal_informal_which, formal_informal_percent=formal_informal_percent)
    new_message.save()
    return HttpResponse(msg)

def getMessages(request, room):
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    messages = Message.objects.filter(room=room_details.id)
    msg = ''

    if (len(messages) > 0) :
        if (messages.values('user').last().get('user') != username) :

            # 존댓말/반말 start
            BASE_DIR = Path(__file__).resolve().parent
                # model_token = os.getenv('MODEL_TOKEN')

            latest_model_path = str(BASE_DIR) + '/saved_model/checkpoint-17754'
            device = 'cpu'
            class FormalClassifier(object):
                def __init__(self):
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        latest_model_path).to(device)
                    self.tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

                def predict(self, text: str):
                    text = clean(text)
                    inputs = self.tokenizer(
                        text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
                    input_ids = inputs["input_ids"].to(device)
                    token_type_ids = inputs["token_type_ids"].to(device)
                    attentsion_mask = inputs["attention_mask"].to(device)

                    model_inputs = {
                        "input_ids": input_ids,
                        "token_type_ids": token_type_ids,
                        "attention_mask": attentsion_mask,
                    }
                    return torch.softmax(self.model(**model_inputs).logits, dim=-1)

                def is_formal(self, text):
                    if self.predict(text)[0][1] > self.predict(text)[0][0]:
                        return True
                    else:
                        return False

                def formal_percentage(self, text):
                    return round(float(self.predict(text)[0][1]), 2)

                def formal_informal_which(self, text):
                    result = self.formal_percentage(text)
                    # print(result)
                    if result > 0.5:
                        formal_informal_which = 'formal'
                    if result < 0.5:
                        formal_informal_which = 'informal'
                    return formal_informal_which
                    
                def formal_informal_percent(self, text):
                    result = self.formal_percentage(text)
                    # print(result)
                    if result > 0.5: # formal
                        formal_informal_percent = result * 100
                    if result < 0.5: # informal
                        formal_informal_percent = result * 100 # 존댓말일 확률
                        # formal_informal_percent = (1 - result) * 100
                    return formal_informal_percent

            message = messages.values('value').last().get('value')
            username = messages.values('user').last().get('user')
            room_id = room
            
            classifier = FormalClassifier()
            formal_informal_which = classifier.formal_informal_which(message)
            formal_informal_percent = classifier.formal_informal_percent(message)

            UserDB = User.objects.values()

            UserFormalInformalDB = UserFormalInformal.objects.values()

            for a in UserFormalInformalDB:
                if a['user'] == username:
                    if a['formal_percent_avg'] >= 80:
                        if formal_informal_which == 'informal':
                            print('이상///////////////////////')
                            msg = "warning !!! \n\n상대방은 지금까지 존댓말을 " + str(a['formal_count']) + "번 사용했으며, " + str(a['formal_percent_avg']) + "%의 확률로 존댓말을 사용할 것으로 예측했지만 반말을 사용하였습니다.\n\n메신저피싱으로 인한 지인 사칭이 아닌지 주의하시길 바랍니다."
                    if a['formal_percent_avg'] <= 20:
                        if formal_informal_which == 'formal':
                            print('이상///////////////////////')
                            msg = "warning !!! \n\n상대방은 지금까지 반말을 " + str(a['informal_count']) + "번 사용했으며, " + str(100 - a['formal_percent_avg']) + "%의 확률로 반말을 사용할 것으로 예측했지만 존댓말을 사용하였습니다.\n\n메신저피싱으로 인한 지인 사칭이 아닌지 주의하시길 바랍니다."
            # 존댓말/반말 end
            


            # 유사단어 체크 start
            okt = Okt()
            nouns = okt.nouns(message)

            father = ['아빠', '아버지', '아부지', '아부이', '아방', '대디']
            mother = ['엄마', '어머니', '어무니', '어무이', '어망', '마미']
            sister = ['언니', '언닝', '온니', '온닝', '언뉘', '온뉘']

            all = father + mother + sister

            for noun in nouns:
                if noun in all:
                    UserDB = User.objects.values()

                    for a in UserDB:
                        if a['count_var'] >= 5:
                            if a['voca'] in father and noun in father:
                                if a['voca'] != noun:
                                    msg = "warning !!!\n\n상대방은 \"" + str(a['voca']) + "\"라는 단어를 " + str(a['count_var']) + "번 사용해왔지만 이번에는 \"" + noun + "\"(이)라는 단어를 사용하였습니다.\n\n메신저피싱으로 인한 지인 사칭이 아닌지 주의하시길 바랍니다."
                            elif a['voca'] in mother and noun in mother:
                                if a['voca'] != noun:
                                    print("이상")
                                    msg = "warning !!!\n\n상대방은 \"" + str(a['voca']) + "\"라는 단어를 " + str(a['count_var']) + "번 사용해왔지만 이번에는 \"" + noun + "\"(이)라는 단어를 사용하였습니다.\n\n메신저피싱으로 인한 지인 사칭이 아닌지 주의하시길 바랍니다."
                            elif a['voca'] in sister and noun in sister:
                                if a['voca'] != noun:
                                    msg = "warning !!!\n\n상대방은 \"" + str(a['voca']) + "\"라는 단어를 " + str(a['count_var']) + "번 사용해왔지만 이번에는 \"" + noun + "\"(이)라는 단어를 사용하였습니다.\n\n메신저피싱으로 인한 지인 사칭이 아닌지 주의하시길 바랍니다."
                                    '''
                                    if Warning.objects.filter(room = room_id, user = username).exists():
                                        voca_warning_count = Warning.objects.get(room = room_id, user = username)
                                        voca_warning_count.voca_warning_count = voca_warning_count.voca_warning_count + 1
                                        voca_warning_count.save()
                                    else:
                                        new_warning = Warning.objects.create(room = room_id, user = username, count_var = 1)
                                        new_warning.save()
                                    '''
            # 유사단어 체크 end

            # warning 체크 
            '''
            formal_warning_count = Warning.objects.get(room = room_id, user = username)
            voca_warning_count = Warning.objects.get(room = room_id, user = username)

            if formal_warning_count.formal_warning_count >= 3:
                msg = "warning !!! \n\n상대방은 지금까지 존댓말을 " + str(a['formal_count']) + "번 사용했으며, " + str(a['formal_percent_avg']) + "%의 확률로 존댓말을 사용할 것으로 예측했지만 반말을 사용하였습니다.\n\n메신저피싱으로 인한 지인 사칭이 아닌지 주의하시길 바랍니다."
            if voca_warning_count.voca_warning_count >= 3:
                msg = "warning !!!\n\n상대방은 " + str(a['voca']) + "라는 단어를 " + str(a['count_var']) + "번 사용해왔지만 이번에는 " + noun + "이라는 단어를 사용하였습니다.\n\n메신저피싱으로 인한 지인 사칭이 아닌지 주의하시길 바랍니다."
            '''                        
  
    return JsonResponse({"messages":list(messages.values()), "alert": msg})

