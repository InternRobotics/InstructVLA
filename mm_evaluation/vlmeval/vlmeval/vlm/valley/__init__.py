try:
    from .valley_eagle_chat import ValleyEagleChat
except:
    from ..base import BaseModel
    class ValleyEagleChat(BaseModel):
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            pass
    # print('valley_eagle_chat not found')
