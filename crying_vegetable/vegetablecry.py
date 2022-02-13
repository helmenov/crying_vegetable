import pandas as pd
import numpy as np

def read_jikken(jikken_tsv,time_prefix,time_unit='second'):
    jikken_df = pd.read_csv(jikken_tsv,header=1,sep='\t',encoding='shift-jis')
    jikken_df = jikken_df[:-1]

    jikken_df = jikken_df[['行ラベル', 'Unnamed: 5']]
    #jikken_df

    # サンプリング時間系列[minute]
    pstr = jikken_df['行ラベル']

    # %%
    # 行ラベルは　prefix + 02%d + .'asd' のフォーマット
    # time_prefixを確認して，下のprefixに記入
    #time_prefix = 'jikken8.200'
    #time_prefix = 'jikken300'


    # %%

    pstr = pstr.str.replace(time_prefix,'',regex=True)
    pstr = pstr.str.replace('.asd','',regex=True)
    t = pstr.values.astype('float')

    time_unit_set = {"days":60*60*24,"hours":60*60,"minutes":60,"seconds":1}
    t = t*time_unit_set[time_unit]    

    # 計測値
    x = jikken_df['Unnamed: 5'].values.astype('float')

    return t,x

def resample_jikken(t,x):
    # 最も短いサンプリング間隔を採用する．
    fs_data0 = 1/np.amin(np.diff(t))
    # 欠損は前後の線形補間で内挿する．
    t2 = np.arange(np.amax(t)*fs_data0)/fs_data0
    x2 = np.full((len(t2),),0.0)
    for i,tt in enumerate(t2):
        if tt in t:
            ii = np.where(tt==t)[0]
            x2[i] = x[ii]
        else:
            r = t[ii+1]-t[ii]
            d = t2[i]-t2[i-1]
            x2[i] = (r-d)/r * x[ii] + d/r * x[ii+1]
    return t2, x2

        