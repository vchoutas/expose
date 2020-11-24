# -*- coding: utf-8 -*-
#


class BoneLinks:

    def __init__(self):
        self.__links = {}
    
    def get(self, bone_name: str, offset=0):
        if bone_name not in self.__links:
            return None
        if offset == 0:
            # オフセットなしの場合、そのまま返す
            return self.__links[bone_name]
        else:
            # オフセットありの場合、その分ずらす
            target_bone_index = self.index(bone_name)
            
            # オフセット加味して、該当INDEXを探す
            for lidx, lkey in enumerate(self.__links.keys()):
                if lidx == target_bone_index + offset:
                    return self.__links[lkey]
        
        return None
    
    def all(self):
        return self.__links

    # リンクに追加
    def append(self, bone):
        self.__links[bone.name] = bone.copy()
    
    # リンクの反転
    def reversed(self):
        return reversed(self.__links)
    
    # リンクの大きさ
    def size(self):
        return len(self.__links.keys())
    
    # 指定されたボーン名までのインデックス
    def index(self, bone_name: str):
        for lidx, lkey in enumerate(self.__links.keys()):
            if lkey == bone_name:
                return lidx
        return -1

    # 指定されたボーン名までのリンクを取得
    def from_links(self, bone_name: str):
        new_links = BoneLinks()
        for lidx, lkey in enumerate(self.__links.keys()):
            new_links.append(self.__links[lkey])
            if lkey == bone_name:
                break
        return new_links

    # 指定されたボーン名以降のリンクを取得
    def to_links(self, bone_name: str):
        new_links = BoneLinks()
        is_append = False
        for lidx, lkey in enumerate(self.__links.keys()):
            if lkey == bone_name:
                is_append = True
            
            if is_append:
                new_links.append(self.__links[lkey])
                
        return new_links
        
    # 最後のリンク名を取得する
    def last_name(self):
        if not self.__links:
            return ""

        return list(self.__links.keys())[-1]
    
    # 最後のリンク名を取得する
    def last_display_name(self):
        if not self.__links:
            return ""

        return list(self.__links.keys())[-1].replace("実体", "")

    # 最初のリンク名を取得する
    def first_name(self):
        if not self.__links:
            return ""

        return list(self.__links.keys())[0]

    # 最初のリンク名を取得する
    def first_display_name(self):
        if not self.__links:
            return ""

        return list(self.__links.keys())[0].replace("実体", "")

    # 指定されたボーン名のみを入れたリンクを取得
    def pickup_links(self, bone_names: list):
        new_links = BoneLinks()
        for lidx, lkey in enumerate(self.__links.keys()):
            if lidx == 0:
                # 末端は常に登録
                new_links.append(self.__links[lkey])
            else:
                if lkey in bone_names:
                    # それ以外はボーン名リストにあること
                    new_links.append(self.__links[lkey])
                
        return new_links
        
    def __str__(self):
        return "<BoneLinks links:{0}".format(self.__links)



