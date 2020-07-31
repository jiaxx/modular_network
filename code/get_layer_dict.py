# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:47:39 2018

@author: Xiaoxuan Jia
"""
# # first estimate from CSD then validate with layer PSTH

# two layer
def get_layer_dict(mouse_ID, layer):

    if mouse_ID=='306046':
        # not very responsive V1
        if layer==2:
            dict_layer={}
            dict_layer['probeA']= [60, 140, 200]
            #dict_layer['probeB']=[60, 87, 200]
            dict_layer['probeC']=[60, 87, 200]
            dict_layer['probeD']=[0, 77, 200]
            dict_layer['probeE']=[0, 64, 200]
            dict_layer['probeF']=[60, 110, 200]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']= [60, 140-8, 140+8, 200]
            dict_layer['probeC']=[60, 87-8, 87+8, 200]
            dict_layer['probeD']=[0, 77-8, 77+8, 200]
            dict_layer['probeE']=[0, 64-8, 64+8, 200]
            dict_layer['probeF']=[60, 110-8, 110+8, 200]

    if mouse_ID=='317944':
        # PV very strong oscillation; better not to include in final dataset
        if layer==2:
            dict_layer={}
            dict_layer['probeA']= [60, 120, 200]
            dict_layer['probeC']=[60, 120, 200]
            dict_layer['probeE']=[60, 121, 200]
            dict_layer['probeF']=[60, 120, 200]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']= [60, 120-8, 120+8, 200]
            dict_layer['probeC']=[60, 120-8, 120+8, 200]
            dict_layer['probeE']=[60, 121-8, 121+8, 200]
            dict_layer['probeF']=[60, 120-8, 120+8, 200]

    if mouse_ID=='326308':
        # VIP in general lower responsiveness; better not to include in final dataset
        if layer==2:
            dict_layer={}
            dict_layer['probeA']= [60, 120, 200]
            dict_layer['probeB']=[60, 75, 200]
            dict_layer['probeC']=[60, 120, 200]
            dict_layer['probeE']=[60, 121, 200]
            dict_layer['probeF']=[60, 120, 200]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']= [60, 120-8,120+8, 200]
            dict_layer['probeB']=[60, 75-8, 75+8, 200]
            dict_layer['probeC']=[60, 120-8, 120+8, 200]
            dict_layer['probeE']=[60, 121-8, 121+8, 200]
            dict_layer['probeF']=[60, 120-8, 120+8, 200]

    # separating into superficial and deep layers
    # manual defination of layers
    if mouse_ID=='388187':
        # not well driven in AM, LM, RL; better not to include in final dataset
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[200, 260, 340]
            dict_layer['probeB']=[60, 154, 260]
            dict_layer['probeC']=[100, 128, 260]
            dict_layer['probeD']=[140, 180, 320]
            dict_layer['probeE']=[150, 248, 380]
            dict_layer['probeF']=[150, 230, 350]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[200, 260-8, 260+8, 340]
            dict_layer['probeB']=[60, 154-8, 154+8, 260]
            dict_layer['probeC']=[100, 128-8, 128+8, 260]
            dict_layer['probeD']=[140, 180-8, 180+8, 320]
            dict_layer['probeE']=[150, 248-8, 248+8, 380]
            dict_layer['probeF']=[150, 230-8, 230+8, 350]

    if mouse_ID=='388523':
        # all areas are driven
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[230, 280, 380]
            dict_layer['probeB']=[200, 275, 380] 
            dict_layer['probeC']=[230, 264, 360]
            dict_layer['probeD']=[200, 250, 360]
            dict_layer['probeE']=[260, 300, 380]
            dict_layer['probeF']=[220, 264, 360]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[230, 280-8, 280+8, 380]
            dict_layer['probeB']=[200, 275-8, 275+8, 380] 
            dict_layer['probeC']=[230, 264-8, 264+8, 360]
            dict_layer['probeD']=[200, 250-8, 250+8, 360]
            dict_layer['probeE']=[260, 300-8, 300+8, 380]
            dict_layer['probeF']=[220, 264-8, 264+8, 360]

    if mouse_ID=='389262':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[230, 300, 380]
            dict_layer['probeB']=[250, 310, 380] 
            dict_layer['probeC']=[210, 277, 380]
            dict_layer['probeD']=[190, 239, 360]
            dict_layer['probeE']=[240, 281, 380]
            dict_layer['probeF']=[210, 273, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[230, 300-8, 300+8, 380]
            dict_layer['probeB']=[250, 310-8, 310+8, 380] 
            dict_layer['probeC']=[210, 277-8, 277+8, 380]
            dict_layer['probeD']=[190, 239-8, 239+8, 360]
            dict_layer['probeE']=[240, 281-8, 281+8, 380]
            dict_layer['probeF']=[210, 273-8, 273+8, 380]

    if mouse_ID=='408153':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 238, 380]
            dict_layer['probeB']=[180, 233, 380] 
            dict_layer['probeC']=[150, 209, 380]
            dict_layer['probeD']=[180, 215, 360]
            dict_layer['probeE']=[220, 288, 380]
            dict_layer['probeF']=[210, 293, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 238-8, 238+8, 380]
            dict_layer['probeB']=[180, 233-8, 233+8, 380] 
            dict_layer['probeC']=[150, 209-8, 209+8, 380]
            dict_layer['probeD']=[180, 215-8, 215+8, 360]
            dict_layer['probeE']=[220, 288-8, 288+8, 380]
            dict_layer['probeF']=[210, 293-8, 293+8, 380]

    if mouse_ID=='408155':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 223, 380]
            dict_layer['probeB']=[180, 240, 380] 
            dict_layer['probeC']=[150, 217, 380]
            dict_layer['probeD']=[180, 207, 360]
            dict_layer['probeE']=[220, 280, 380]
            dict_layer['probeF']=[210, 261, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 223-8, 223+8, 380]
            dict_layer['probeB']=[180, 240-8, 240+8, 380] 
            dict_layer['probeC']=[150, 217-8, 217+8, 380]
            dict_layer['probeD']=[180, 207-8, 207+8, 360]
            dict_layer['probeE']=[220, 280-8, 280+8, 380]
            dict_layer['probeF']=[210, 261-8, 261+8, 380]

    if mouse_ID=='410344':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 223, 380]
            dict_layer['probeB']=[180, 216, 380] 
            dict_layer['probeC']=[150, 193, 380]
            dict_layer['probeD']=[180, 192, 360]
            dict_layer['probeE']=[220, 246, 380]
            dict_layer['probeF']=[210, 253, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 223-8, 223+8, 380]
            dict_layer['probeB']=[180, 216-8, 216+8, 380] 
            dict_layer['probeC']=[150, 193-8, 193+8, 380]
            dict_layer['probeD']=[180, 192-8, 192+8, 360]
            dict_layer['probeE']=[220, 246-8, 246+8, 380]
            dict_layer['probeF']=[210, 253-8, 253+8, 380]

    if mouse_ID=='412796':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 240, 380]
            dict_layer['probeB']=[180, 254, 380] 
            dict_layer['probeC']=[130, 201, 380]
            dict_layer['probeD']=[180, 230, 360]
            dict_layer['probeE']=[220, 294, 380]
            dict_layer['probeF']=[210, 254, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 240-8, 240+8, 380]
            dict_layer['probeB']=[180, 254-8, 254+8, 380] 
            dict_layer['probeC']=[150, 201-8, 201+8, 380]
            dict_layer['probeD']=[180, 230-8, 230+8, 360]
            dict_layer['probeE']=[220, 294-8, 294+8, 380]
            dict_layer['probeF']=[210, 254-8, 254+8, 380]

    if mouse_ID=='412804':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 266, 380]
            dict_layer['probeB']=[180, 268, 380] 
            dict_layer['probeC']=[150, 232, 380]
            dict_layer['probeD']=[160, 217, 380]
            dict_layer['probeE']=[240, 272, 380]
            dict_layer['probeF']=[250, 298, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 266-8, 266+8, 380]
            dict_layer['probeB']=[180, 268-8, 268+8, 380] 
            dict_layer['probeC']=[150, 232-8, 232+8, 380]
            dict_layer['probeD']=[160, 217-8, 217+8, 380]
            dict_layer['probeE']=[240, 272-8, 272+8, 380]
            dict_layer['probeF']=[250, 298-8, 298+8, 380]

    if mouse_ID=='412809':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[200, 238, 380]
            dict_layer['probeB']=[220, 262, 380] 
            dict_layer['probeC']=[160, 232, 380]
            dict_layer['probeD']=[140, 209, 380]
            dict_layer['probeE']=[200, 256, 380]
            dict_layer['probeF']=[240, 281, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[200, 238-8, 238+8, 380]
            dict_layer['probeB']=[220, 262-8, 262+8, 380] 
            dict_layer['probeC']=[160, 232-8, 232+8, 380]
            dict_layer['probeD']=[140, 209-8, 209+8, 380]
            dict_layer['probeE']=[200, 256-8, 256+8, 380]
            dict_layer['probeF']=[240, 281-8, 281+8, 380]

    if mouse_ID=='415148':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[160, 211, 380]
            dict_layer['probeB']=[160, 202, 380] 
            dict_layer['probeC']=[160, 214, 380]
            dict_layer['probeD']=[160, 222, 380]
            dict_layer['probeE']=[200, 274, 380]
            dict_layer['probeF']=[160, 240, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 211-8, 211+8, 380]
            dict_layer['probeB']=[180, 202-8, 202+8, 380] 
            dict_layer['probeC']=[160, 214-8, 214+8, 380]
            dict_layer['probeD']=[140, 222-8, 222+8, 380]
            dict_layer['probeE']=[200, 274-8, 274+8, 380]
            dict_layer['probeF']=[250, 240-8, 240+8, 380]

    if mouse_ID=='415149':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 262, 380]
            dict_layer['probeB']=[180, 262, 380] 
            dict_layer['probeC']=[160, 232, 380]
            dict_layer['probeD']=[140, 233, 380]
            dict_layer['probeE']=[200, 288, 380]
            dict_layer['probeF']=[250, 324, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 262-8, 262+8, 380]
            dict_layer['probeB']=[180, 262-8, 262+8, 380] 
            dict_layer['probeC']=[160, 232-8, 232+8, 380]
            dict_layer['probeD']=[140, 233-8, 233+8, 380]
            dict_layer['probeE']=[200, 288-8, 288+8, 380]
            dict_layer['probeF']=[250, 324-8, 324+8, 380]

    if mouse_ID=='416356':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 236, 380]
            dict_layer['probeB']=[180, 260, 380] 
            dict_layer['probeC']=[200, 254, 380]
            dict_layer['probeD']=[180, 246, 380]
            dict_layer['probeE']=[233, 288, 380]
            dict_layer['probeF']=[221, 259, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 236-8, 236+8, 380]
            dict_layer['probeB']=[180, 260-8, 260+8, 380] 
            dict_layer['probeC']=[160, 254-8, 254+8, 380]
            dict_layer['probeD']=[160, 246-8, 246+8, 380]
            dict_layer['probeE']=[200, 288-8, 288+8, 380]
            dict_layer['probeF']=[200, 259-8, 259+8, 380]

    if mouse_ID=='416357':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 258, 380]
            dict_layer['probeB']=[180, 267, 380] 
            dict_layer['probeC']=[160, 229, 380]
            dict_layer['probeD']=[160, 250, 380]
            dict_layer['probeE']=[200, 296, 380]
            dict_layer['probeF']=[200, 300, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 258-8, 258+8, 380]
            dict_layer['probeB']=[180, 267-8, 267+8, 380] 
            dict_layer['probeC']=[160, 229-8, 229+8, 380]
            dict_layer['probeD']=[160, 250-8, 250+8, 380]
            dict_layer['probeE']=[200, 296-8, 296+8, 380]
            dict_layer['probeF']=[200, 300-8, 300+8, 380]

    if mouse_ID=='416856':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 219, 380]
            dict_layer['probeB']=[180, 209, 380] 
            dict_layer['probeC']=[160, 182, 380]
            dict_layer['probeD']=[160, 169, 380]
            dict_layer['probeE']=[200, 229, 380]
            dict_layer['probeF']=[200, 282, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 219-8, 219+8, 380]
            dict_layer['probeB']=[180, 209-8, 209+8, 380] 
            dict_layer['probeC']=[160, 182-8, 182+8, 380]
            dict_layer['probeD']=[160, 169-8, 169+8, 380]
            dict_layer['probeE']=[200, 229-8, 229+8, 380]
            dict_layer['probeF']=[200, 282-8, 282+8, 380]

    if mouse_ID=='416861':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[160, 210, 380]
            dict_layer['probeB']=[180, 209, 380] 
            dict_layer['probeC']=[160, 190, 380]
            dict_layer['probeD']=[140, 188, 380]
            dict_layer['probeE']=[200, 248, 380]
            dict_layer['probeF']=[200, 271, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 210-8, 210+8, 380]
            dict_layer['probeB']=[180, 209-8, 209+8, 380] 
            dict_layer['probeC']=[160, 190-8, 190+8, 380]
            dict_layer['probeD']=[160, 188-8, 188+8, 380]
            dict_layer['probeE']=[200, 248-8, 248+8, 380]
            dict_layer['probeF']=[200, 271-8, 271+8, 380]

    if mouse_ID=='419112':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[160, 230, 380]
            dict_layer['probeB']=[220, 264, 380] 
            dict_layer['probeC']=[220, 274, 380]
            dict_layer['probeD']=[200, 253, 380]
            dict_layer['probeE']=[220, 294, 380]
            dict_layer['probeF']=[220, 296, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[160, 230-8, 230+8, 380]
            dict_layer['probeB']=[220, 264-8, 264+8, 380] 
            dict_layer['probeC']=[220, 274-8, 274+8, 380]
            dict_layer['probeD']=[200, 253-8, 253+8, 380]
            dict_layer['probeE']=[200, 294-8, 294+8, 380]
            dict_layer['probeF']=[200, 296-8, 296+8, 380]

    if mouse_ID=='419114':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[160, 191, 380]
            dict_layer['probeB']=[160, 184, 380] 
            dict_layer['probeC']=[140, 163, 380]
            dict_layer['probeD']=[160, 157, 380]
            dict_layer['probeE']=[200, 221, 380]
            dict_layer['probeF']=[200, 250, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[160, 191-8, 191+8, 380]
            dict_layer['probeB']=[160, 184-8, 184+8, 380] 
            dict_layer['probeC']=[140, 163-8, 163+8, 380]
            dict_layer['probeD']=[160, 157-8, 157+8, 380]
            dict_layer['probeE']=[200, 221-8, 221+8, 380]
            dict_layer['probeF']=[200, 250-8, 250+8, 380]

    if mouse_ID=='419115':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[120, 173, 380]
            dict_layer['probeB']=[120, 168, 380] 
            dict_layer['probeC']=[120, 140, 380]
            dict_layer['probeD']=[120, 149, 380]
            dict_layer['probeE']=[160, 213, 380]
            dict_layer['probeF']=[160, 242, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[120, 173-8, 173+8, 380]
            dict_layer['probeB']=[120, 168-8, 168+8, 380] 
            dict_layer['probeC']=[120, 140-8, 140+8, 380]
            dict_layer['probeD']=[120, 149-8, 149+8, 380]
            dict_layer['probeE']=[160, 213-8, 213+8, 380]
            dict_layer['probeF']=[160, 242-8, 242+8, 380]

    if mouse_ID=='419116':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[120, 205, 380]
            dict_layer['probeB']=[120, 220, 380] 
            dict_layer['probeC']=[120, 218, 380]
            dict_layer['probeD']=[120, 207, 380]
            dict_layer['probeE']=[160, 244, 380]
            dict_layer['probeF']=[160, 259, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[120, 173-8, 173+8, 380]
            dict_layer['probeB']=[120, 168-8, 168+8, 380] 
            dict_layer['probeC']=[120, 140-8, 140+8, 380]
            dict_layer['probeD']=[120, 149-8, 149+8, 380]
            dict_layer['probeE']=[160, 213-8, 213+8, 380]
            dict_layer['probeF']=[160, 242-8, 242+8, 380]

    if mouse_ID=='419117':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 260, 380]
            dict_layer['probeB']=[180, 251, 380] 
            dict_layer['probeC']=[160, 213, 380]
            dict_layer['probeD']=[160, 216, 380]
            dict_layer['probeE']=[180, 259, 380]
            dict_layer['probeF']=[200, 299, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 260-8, 260+8, 380]
            dict_layer['probeB']=[180, 251-8, 251+8, 380] 
            dict_layer['probeC']=[160, 213-8, 213+8, 380]
            dict_layer['probeD']=[160, 216-8, 216+8, 380]
            dict_layer['probeE']=[180, 259-8, 259+8, 380]
            dict_layer['probeF']=[200, 299-8, 299+8, 380]

    if mouse_ID=='419118':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[180, 236, 380]
            dict_layer['probeB']=[180, 243, 380] 
            dict_layer['probeC']=[160, 223, 380]
            dict_layer['probeD']=[140, 186, 380]
            dict_layer['probeE']=[180, 256, 380]
            dict_layer['probeF']=[200, 265, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[180, 236-8, 236+8, 380]
            dict_layer['probeB']=[180, 243-8, 243+8, 380] 
            dict_layer['probeC']=[160, 223-8, 223+8, 380]
            dict_layer['probeD']=[140, 186-8, 186+8, 380]
            dict_layer['probeE']=[180, 256-8, 256+8, 380]
            dict_layer['probeF']=[200, 265-8, 265+8, 380]

    if mouse_ID=='419119':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[160, 224, 380]
            dict_layer['probeB']=[180, 262, 380] 
            dict_layer['probeC']=[160, 240, 380]
            dict_layer['probeD']=[140, 235, 380]
            dict_layer['probeE']=[180, 272, 380]
            dict_layer['probeF']=[200, 276, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[160, 224-8, 224+8, 380]
            dict_layer['probeB']=[180, 262-8, 262+8, 380] 
            dict_layer['probeC']=[160, 240-8, 240+8, 380]
            dict_layer['probeD']=[140, 235-8, 235+8, 380]
            dict_layer['probeE']=[180, 272-8, 272+8, 380]
            dict_layer['probeF']=[200, 276-8, 276+8, 380]

    if mouse_ID=='424445':
        # all areas are driven to certain degree
        if layer==2:
            dict_layer={}
            dict_layer['probeA']=[160, 182, 380]
            dict_layer['probeB']=[180, 241, 380] 
            dict_layer['probeC']=[160, 222, 380]
            dict_layer['probeD']=[140, 184, 380]
            dict_layer['probeE']=[180, 256, 380]
            dict_layer['probeF']=[200, 284, 380]
        if layer==3:
            dict_layer={}
            dict_layer['probeA']=[160, 182-8, 182+8, 380]
            dict_layer['probeB']=[180, 241-8, 241+8, 380] 
            dict_layer['probeC']=[160, 222-8, 222+8, 380]
            dict_layer['probeD']=[140, 184-8, 184+8, 380]
            dict_layer['probeE']=[180, 256-8, 256+8, 380]
            dict_layer['probeF']=[200, 284-8, 284+8, 380]



    return dict_layer

