#! /usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import findSystemFonts, FontProperties

def find_times_font(bold=False, italic=False):
    fonts = findSystemFonts()
    for fontpath in fonts:
        fprop = FontProperties(fname=fontpath)
        name = fprop.get_name()
        name_matched = 'Times New Roman' in name
        pname = os.path.splitext(os.path.basename(fontpath))[0]
        style_matched = ((not bold) or (bold and (pname.endswith('Bold') or (pname.lower() == pname and pname.endswith('bd'))))) and \
                        ((not italic) or (italic and (pname.endswith('Italic') or (pname.lower() == pname and pname.endswith('i')))))
        if name_matched and style_matched:
            return fprop
    return None

def cdf_from_histogram(index, counts):
    cum = np.cumsum(counts)
    cdf = cum / np.max(cum)
    return pd.DataFrame(data=cdf, index=index)
