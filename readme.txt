國立屏東科技大學 資管管理系 三A
(1112)影像處理概論(4072) 蔡正發教授
期末專題：
組員：
    B10956065 陳鑫彰
    B10956026 謝東翰
    B10956050 謝心妍
    B10956053 陳榆靜

20230615 1148 66def
20230615 1343 73def

 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

初始化流程:
1. import
    1.1 tkinter(*, ttk), PIL => GUI基礎套件
    1.2 tkinter.filedialog, xml => 附加功能需求
    1.3 cv2, rembg, numpy => 影像處理需求
2. load def
3. load xml file and decoding
    3.1 lang/*.xml
        3.1.1 t = dict() = common text
        3.1.2 n = dict() = scale title
    3.2 data.xml
        3.2.1 f = dict() = function setting(scale value)
4. tkinter init
    4.1 root, mainframe
    4.2 information row
    4.3 menu, menubar, load function
    4.4 scale
    4.5 image frame, image init
        4.5.1 build and grid image frame
        4.5.2 image = theImage() => initialization the most important variable in the project
        4.5.3 call update_image() to update screen
5. __name__ == '__main__': root.mainloop() => 開始tkinter主迴圈

 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

執行流程:
壹、
0. 使用者利用 menubar 切換功能
1. callback_menu()
    1.1 global flag_program_type 並更新應執行之功能
    1.2 更新 information row(目前功能)
    1.3 更新各滑桿之設定(title, value)
2. callback_scale() 更新畫面

貳、
0. 使用者拉動滑桿
1. callback_scale()
    1.1 彙整各滑桿的數值
    1.2 取整並更新介面
    1.3 ↑↓←→
2. image_program()
    2.1 抓取 flag_program_type 確認應執行的功能
    2.2 利用 getattr 確認所對應之def
    2.3 呼叫def並取得回傳值
    2.4 ↑
3. update_image()
    3.1 進行資料型別之轉換
    3.2 更新顯示之圖片

參、
0. 使用者要求讀檔
1. callback_menu_file_load()
    1.1 呼叫 tk.filedialog 確認欲讀檔之位置
    1.2 進行資料型別之轉換
    1.3 ↑
2. theImage.reload()
    2.1 更新原始圖片
    2.2 若新圖片太大則將其縮小
    2.3 呼叫 update_image() 更新畫面(原始圖片)
    2.4 ↑ theImage.reset()
        2.4.1 更新 編輯後圖片
        2.4.2 更新 information row(圖片大小)
    2.5 呼叫 callback_scale() 更新畫面(編輯後圖片)

肆、
0. 使用者要求存檔
1. callback_menu_file_save()
    1.1 呼叫 tk.filedialog 確認欲存檔之位置
    1.2 呼叫 callback_scale() 並取得回傳值
    1.3 進行資料型別之轉換後存檔
