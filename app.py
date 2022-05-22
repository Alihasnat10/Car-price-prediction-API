import pickle
import flask
import io
import string
import time
import os
import numpy as np
from flask import Flask, jsonify, request, render_template
#from machine_learning import car_price_prediction
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_ngrok import run_with_ngrok
import h5py
from tensorflow import keras
loaded_model = keras.models.load_model("model.h5")
app = Flask(__name__)

#run_with_ngrok(app)

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def data_encoding(df):
    df['mileageFromOdometer'] = df['mileageFromOdometer'].replace('[^0-9]+', '', regex=True)
    df['engDisp'] = df['engDisp'].replace('[^0-9]+', '', regex=True)
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset=["engDisp"], inplace=True)
    # converting strings to numberical values
    df['mileageFromOdometer'] = df['mileageFromOdometer'].astype(float)
    df['engDisp'] = df['engDisp'].astype(float)
    df['modelDate'] = df['modelDate'].astype(float)
    # df['extraFeatures.Assembly'].replace({'Local': 0, 'Imported': 1}, inplace=True)
    df['fuelType'].replace({'Petrol': 0, 'Diesel': 1, 'Hybrid': 2, 'CNG': 3, 'Lpg': 4}, inplace=True)
    df['vehicleTransmission'].replace({'Manual': 0, 'Automatic': 1}, inplace=True)
    df['color'].replace(
        {' Urban Titanium': 0, ' color urban Titanium Metallic, Honda ,': 1, '-': 2, '.': 3, '...': 4, '0': 5,
         '14/ 16': 6, '3 shaded german colour': 7, 'A.Silver': 8, 'AQ Jade': 9, 'AQ.JADE': 10, 'AQUA BLUE': 11,
         'AS YOU CAN SEE': 12, 'AUTOMATIC ': 13, 'Ac installed working': 14, 'All Colors': 15, 'All colors': 16,
         'All colurs': 17, 'Angori': 18, 'Any color required': 19, 'Any colour ': 20, 'Aqua Blue': 21,
         'Aqua Marine': 22, 'Aqua Silver': 23, 'Aqua Sliver': 24, 'Aqua blue': 25, 'Aqua green': 26, 'Aqua green ': 27,
         'Aqua green color': 28, 'Army ': 29, 'Army green': 30, 'Ash White': 31, 'Attitude Black': 32, 'BBPAIRAL': 33,
         'BEIGE': 34, 'BLACK': 35, 'BLUE': 36, 'BLUISH GREEN MICA': 37, 'BOLD BEIGE MATALLIC': 38, 'BRONZE': 39,
         'BURGUNDY': 40, 'BY6': 41, 'Bage': 42, 'Barolo Black': 43, 'Beige': 44, 'Beigh': 45, 'Beigie ICI': 46,
         'Bhatti colour': 47, 'Black': 48, 'Black Pearl': 49, 'Blonde matelic': 50, 'Blue': 51, 'Blue Pearl': 52,
         'Blue silver': 53, 'Blueish mica': 54, 'Blueish silver': 55, 'Blueish silver mica': 56, 'Bluish Silver': 57,
         'Bluish Silver ': 58, 'Bluish Silver Mica Metallica': 59, 'Bluish silver mica metall': 60,
         'Bluvish Silver': 61, 'Bold Beige Metallic': 62, 'Bold beige metallic': 63, 'Bottle Green': 64,
         'Bottle green': 65, 'British': 66, 'British Green': 67, 'British green': 68, 'Bronce Metallic': 69,
         'Bronze': 70, 'Bronze Mica': 71, 'Bronze Mica Metallic': 72, 'Brown': 73, 'Brown Metallic': 74,
         'Brownce mica': 75, 'Burgundy': 76, 'C green': 77, 'C. G Pearl': 78, 'CANYON STONE SILVER': 79,
         'COOL VIOLE': 80, 'CYPRUS BLUE': 81, 'Camel': 82, 'Cappuccino Brown': 83, 'Cashmere Silver': 84,
         'Charcoal grey pearl': 85, 'Cheery Black': 86, 'Chocolate brown': 87, 'Coffee': 88, 'Coffee Brown Color': 89,
         'Cool Bebige': 90, 'Cool Beige': 91, 'Copper pink': 92, 'Crystal Blue': 93, 'Crystal black pearl': 94, 'D': 95,
         'DX Gray': 96, 'Dark Blue': 97, 'Dark Brown': 98, 'Dark Green': 99, 'Dark Grey': 100,
         'Dark Grey colour (original )': 101, 'Dark Mettelic green': 102, 'Dark Purple ': 103, 'Dark blue': 104,
         'Dark gray': 105, 'Deep Red metallic': 106, 'Dorado Gold': 107, 'Electric Blue': 108, 'Electric Blue ': 109,
         'Electric Green': 110, 'Electric Sports Blue': 111, 'Elegant Dark Grey': 112, 'Eminent Blue': 113,
         'Eminent Blue ': 114, 'Eminent blue': 115, 'Emint Blue': 116, 'Evo': 117, 'Ferozi': 118, 'FewTouch Up': 119,
         'Flexin golden': 120, 'Foan': 121, 'G v matell': 122, 'G.V.METTAL': 123, 'GOLD': 124, 'GR VL MATL': 125,
         'GREEN': 126, 'GREEN-van': 127, 'GREY': 128, 'GRPHT grey': 129, 'GUN METALLIC': 130,
         'Gad color khyber suzhuki for sale ': 131, 'Ganmtc': 132, 'Gold': 133, 'Gold ': 134, 'Golden': 135,
         'Golden Brown ': 136, 'Golden Green': 137, 'Golden becb ': 138, 'Golden green': 139, 'Good condition': 140,
         'Granite Metallic': 141, 'Grape Golden': 142, 'Graphite Grey': 143, 'Graphite grey': 144,
         'Graphite grey ': 145, 'Gray': 146, 'Gray dark': 147, 'Graynite Grey': 148, 'Green': 149,
         'Green Black Metalic': 150, 'Green Mat ': 151, 'Green Matelic': 152, 'Green mattalic': 153, 'Greenish': 154,
         'Greenish Grey': 155, 'Grey': 156, 'Grey Metalic': 157, 'Grey Metallic': 158, 'Grey Metallic ': 159,
         'Grey Mettalic': 160, 'Greyphite': 161, 'Grinish': 162, 'Gu matalic': 163, 'Gun . mettalic': 164,
         'Gun Matalic': 165, 'Gun Matelic': 166, 'Gun Matellic': 167, 'Gun Mattalic': 168, 'Gun Metalic': 169,
         'Gun Metallic': 170, 'Gun Metallic ': 171, 'Gun Metallic Colour': 172, 'Gun matalic': 173, 'Gun matelic': 174,
         'Gun matlic': 175, 'Gun mattalic': 176, 'Gun metal Grey': 177, 'Gun metalic': 178, 'Gun metallic': 179,
         'Gun metallic ': 180, 'Gun metelic': 181, 'Gun mettalic': 182, 'Gunmatalic': 183, 'Gunmatlik': 184,
         'Gunmattalic': 185, 'Gunmetallic': 186, 'Gunmettalic': 187, 'Herbarino Red': 188, 'Ice Blue': 189,
         'Ice blue': 190, 'Indica': 191, 'Indigo': 192, 'Inkish Metallic Blue': 193, 'Kahi Green': 194,
         'Keen gold': 195, 'L Green': 196, 'Lavender': 197, 'Lavinder': 198, 'Light Blue': 199, 'Light Gold': 200,
         'Light Golden': 201, 'Light Green': 202, 'Light Purple': 203, 'Light Purple metallic': 204,
         'Light Rose colour': 205, 'Light Shimmering Green': 206, 'Light blue': 207, 'Light green': 208,
         'Light green ': 209, 'Light grey': 210, 'Light metallic green': 211, 'Light pink': 212, 'Lilac Blue': 213,
         'Liliac blue': 214, 'Liquid blue': 215, 'Lite green': 216, 'Lunar Silver': 217, 'MAGNETA': 218, 'MAROON': 219,
         'METALLIC FLAXEN MICA': 220, 'Magenta': 221, 'Magneta': 222, 'Malti': 223, 'Manhattan Gray': 224,
         'Maroon': 225, 'Maroon Most beautiful color': 226, 'Matalic': 227, 'Matalic Green': 228, 'Matel Rose ': 229,
         'Matelic': 230, 'Matellic Grey': 231, 'Medium Silver': 232, 'Mehendi': 233, 'Mercury Blue': 234,
         'Metal Metallic': 235, 'Metal metallic': 236, 'Metalic': 237, 'Metalic Blue': 238, 'Metalic Green': 239,
         'Metalic blue': 240, 'Metalic flax': 241, 'Metallic': 242, 'Metallic ': 243, 'Metallic Blue': 244,
         'Metallic Golden': 245, 'Metallic Gray': 246, 'Metallic Green': 247, 'Metallic Grey': 248,
         'Metallic Grey/Green': 249, 'Metallic Silver': 250, 'Metallic blue': 251, 'Metallic green': 252,
         'Metallic grey': 253, 'Metallic steel': 254, 'Mettalic': 255, 'Mettalic Green': 256, 'Midnight Blue': 257,
         'Military Green': 258, 'Mint Blue': 259, 'Modern Steel': 260, 'Modern steel': 261, 'Modren steel': 262,
         'Moon Light': 263, 'Morpho Blue Pearl': 264, 'Motelic grey': 265, 'Mud Silver': 266, 'Mulberry Black': 267,
         'Multi': 268, 'Mustard': 269, 'N': 270, 'NIL': 271, 'Nato Green': 272, 'Navy': 273, 'Navy Blue': 274,
         'Navy blue': 275, 'No idea': 276, 'Not Defined': 277, 'OFfwhite': 278, 'Ocean mist metallic': 279,
         'Oliva meca metellic': 280, 'Olive': 281, 'Olive ': 282, 'Olive Green': 283,
         'Olive Green / Metallic Green': 284, 'Olive green': 285, 'Olive metallic ': 286, 'Orange': 287,
         'Own color': 288, 'PEARL WHITE': 289, 'PEARL-WHITE': 290, 'PEARLWHITE': 291, 'PINK': 292,
         'Panthera metal': 293, 'Panthera metallic black': 294, 'Peach': 295, 'Peal White': 296, 'Pearl': 297,
         'Pearl Black': 298, 'Pearl Brinjal': 299, 'Pearl Maroon': 300, 'Pearl Purple ': 301,
         'Pearl Sky Magnisum ': 302, 'Pearl Tea Pink': 303, 'Pearl White': 304, 'Pearl White ': 305, 'Pearl black': 306,
         'Pearl white': 307, 'Pearl white ': 308, 'Perl White': 309, 'Perl white': 310, 'Perl white ': 311,
         'Phantom brown': 312, 'Pink': 313, 'Pink Rose': 314, 'Polish Matilic': 315, 'Polish Metalic': 316,
         'Polish Metallic': 317, 'Polished Metallic Grey': 318, 'Pu': 319, 'Purl white ': 320, 'Purple': 321,
         'RED': 322, 'RED PEARL': 323, 'RED WINE': 324, 'REDVINE': 325, 'Red': 326, 'Red Pearl': 327, 'Red Vien': 328,
         'Red Vine': 329, 'Red Wine': 330, 'Red Wine ': 331, 'Red vaine': 332, 'Red vine': 333, 'Red wine': 334,
         'Red wine ': 335, 'Red wyne': 336, 'Redwine': 337, 'Redwine Metallic': 338, 'Rose ': 339, 'Rose Black': 340,
         'Rose Gold': 341, 'Rose Maitlak': 342, 'Rose Meta': 343, 'Rose Metallic': 344, 'Rose Metallic ': 345,
         'Rose Mist': 346, 'Rose Pink': 347, 'Rose Silver': 348, 'Rose colour': 349, 'Rose gold': 350,
         'Rose matalik': 351, 'Rose metal': 352, 'Rose metallic': 353, 'Rose metallic color': 354, 'Rose mint': 355,
         'Rose mist': 356, 'Rose pink': 357, 'Rosemist': 358, 'Rost Mist': 359, 'Royal Blue': 360, 'Royal white': 361,
         'S': 362, 'S Green': 363, 'S Smoke GR': 364, 'S.R metelic': 365, 'SILVER': 366, 'SKY BLUE': 367, 'Salon': 368,
         'Saloon': 369, 'Same as pic': 370, 'Samoke Green': 371, 'Sand Beidge': 372, 'Sand Beige': 373,
         'Sand biege': 374, 'Sea Blue ': 375, 'Sea Green': 376, 'Sea Green ': 377, 'Sea green': 378, 'SeaGreen': 379,
         'Shade': 380, 'Shalimar': 381, 'Shalimar Gold, Rose Pink': 382, 'Shalimar Mist': 383, 'Shalimar Rose': 384,
         'Shalimar Rose Mist': 385, 'Shalimar rose': 386, 'Shalimar rose color': 387, 'Shalimar rose mist': 388,
         'Shemreen Green': 389, 'Shemring Green': 390, 'Shimer Green': 391, 'Shimer green': 392, 'Shimiring Green': 393,
         'Shimmering Green Metalic': 394, 'Shimmering Green Metalltc': 395, 'Shimmering Green Silver': 396,
         'Shimmering green': 397, 'Shimmering green metallic': 398, 'Shimreen Green': 399, 'Shimrin Green': 400,
         'Shining Pearl White': 401, 'Shrimp green': 402, 'Silver': 403, 'Silver Cream': 404, 'Silver Graphite': 405,
         'Silver Green': 406, 'Silver Pink': 407, 'Silver sky': 408, 'Silvergolden': 409, 'Skin': 410, 'Sky Blue': 411,
         'Sky Blue ': 412, 'Sky Light Blue': 413, 'Sky blue': 414, 'Sky blue ': 415, 'Skyblue': 416, 'Smo Grey': 417,
         'Smoke Gr': 418, 'Smoke Green': 419, 'Smoke Green ': 420, 'Smoke Grey': 421, 'Smoke gray': 422,
         'Smoke green': 423, 'Smokey Green': 424, 'Snoke green': 425, 'Steel Metallic': 426, 'Strong Blue': 427,
         'Strong Metallic Blue': 428, 'Super Dark Blue': 429, 'Super White': 430, 'Super white': 431,
         'Super white ': 432, 'SuperWhite': 433, 'T': 434, 'T-Green': 435, 'TURQOUISE': 436, 'Taffeta white': 437,
         'Tay Deep': 438, 'Tea Pink': 439, 'Titainum': 440, 'Titanium': 441, 'Titanium ': 442, 'Titanium Brown': 443,
         'Titanium metalic ': 444, 'Titanium red': 445, 'Titilium mettalic': 446, 'Trim': 447, 'Turquoise': 448,
         'Turquoise blue': 449, 'Two tone silver and green': 450, 'Type K DK-Grey': 451, 'UNLISTED': 452,
         'URBAN TITANIUM': 453, 'URBAN TITANIUM ': 454, 'Unban Tatian': 455, 'Unlisted': 456, 'Unlosted': 457,
         'Urban Titanium': 458, 'Urban Titanium ': 459, 'Urban Titanium Metallic': 460, 'Urban Titenium': 461,
         'Urban grey': 462, 'Urban titanium': 463, 'Urban titanium ': 464, 'Urban titanum': 465, 'Verde Metallic': 466,
         'Vine Red': 467, 'Violet Mist Metallic': 468, 'WHITE': 469, 'WHITE WITH BEIGE AND BLACK WITH RED': 470,
         'WINE RED': 471, 'Warm Silver': 472, 'White': 473, 'Wine': 474, 'Wine Red': 475, 'Wine red': 476,
         'Yellow': 477, 'a': 478, 'amint blue': 479, 'angori': 480, 'angori coulor': 481, 'antique': 482,
         'aqua blue': 483, 'aqua green': 484, 'bache': 485, 'beige and green': 486, 'beige matalic': 487,
         'bhati color': 488, 'biscuit': 489, 'biscut': 490, 'black n yellow': 491, 'black with gun metallic': 492,
         'blackish green shade': 493, 'blackish grew': 494, 'blackish nevy blue peral': 495, 'blue metallic': 496,
         'blue type': 497, 'blueish green pearl': 498, 'blueish silver': 499, 'bluewish': 500, 'bluish purple': 501,
         'bluish silver': 502, 'bold matalic beig': 503, 'boron': 504, 'bottle Green': 505, 'bottle green': 506,
         'brown metallic': 507, 'brownish black': 508, 'bule mint': 509, 'c Green': 510, 'c green': 511,
         'canuon stone silver': 512, 'cashmare silver': 513, 'cegreen': 514, 'champagane': 515, 'check pictures': 516,
         'coffee': 517, 'color': 518, 'color urban Titanium Metallic': 519, 'cool Red': 520, 'cool beige': 521,
         'cool red': 522, 'cream': 523, 'cream ': 524, 'cream color': 525, 'cypres blue': 526, 'cyprus blue': 527,
         'dalmatalak': 528, 'dard colour': 529, 'dark brown': 530, 'dark green': 531, 'dark grey': 532,
         'data migration': 533, 'dim golden': 534, 'drk green': 535, 'e.g metallic color ': 536, 'electric blue': 537,
         'executive colour': 538, 'fixen mica': 539, 'fontain brown': 540, 'full shower': 541, 'g gray': 542,
         'gan matalik': 543, 'gan matelk': 544, 'gary': 545, 'gold white': 546, 'gold wight': 547, 'golden': 548,
         'golden hai': 549, 'golden shalimar': 550, 'golden silke': 551, 'golden silver': 552, 'golden type': 553,
         'golden white': 554, 'goldensilver': 555, 'good': 556, 'gra': 557, 'grafyat grey': 558,
         'granite metallic': 559, 'graphite grey': 560, 'grawish': 561, 'gray': 562, 'gray graphite': 563,
         'gray green colour': 564, 'gre gun': 565, 'green dark': 566, 'green matalic': 567, 'green mattalic': 568,
         'green metallic': 569, 'greenish': 570, 'greenish blue': 571, 'greenish type': 572, 'grefite': 573,
         'greish green': 574, 'gresh green': 575, 'grey blue type': 576, 'grey brown': 577, 'grey dark': 578,
         'grey green': 579, 'grey greenish': 580, 'grey metallic': 581, 'grey mettalic': 582, 'greyish purple': 583,
         'grin': 584, 'grinish': 585, 'grmit': 586, 'gun .matalic': 587, 'gun matalic': 588, 'gun matelic': 589,
         'gun matellic': 590, 'gun matlic': 591, 'gun mattalic': 592, 'gun mattelic': 593, 'gun metal': 594,
         'gun metalic': 595, 'gun metallic': 596, 'gun metallic ': 597, 'gun metallic colour': 598, 'gun metlic': 599,
         'gun mettalic': 600, 'gun mtalic': 601, 'gun mtalic clr': 602, 'gun mtalick': 603, 'gun mtelic': 604,
         'gun mutalic': 605, 'gunmatalic': 606, 'gunmatellic': 607, 'gunmatlic': 608, 'gunmatlick': 609,
         'gunmetalic': 610, 'ice blue': 611, 'illusion grey': 612, 'inner steel grey, outer green': 613, 'jeniu': 614,
         'jett grey': 615, 'lavender': 616, 'leaf Green Smooth Colour': 617, 'lemon': 618, 'lighat green': 619,
         'light': 620, 'light Blue': 621, 'light Green': 622, 'light Pink': 623, 'light blue': 624, 'light brown': 625,
         'light brown like beige colour': 626, 'light golden': 627, 'light green': 628, 'light gren': 629,
         'light perpul': 630, 'light pink': 631, 'light sky': 632, 'light sky blue': 633, 'lightly blue': 634,
         'lilac': 635, 'lite golden': 636, 'lite green': 637, 'lite grey': 638, 'loyal blue': 639, 'm grey': 640,
         'mat green': 641, 'matalic': 642, 'matalic blue': 643, 'matalic colour': 644, 'matelic': 645,
         'matelic blue': 646, 'matelic grey': 647, 'matelic rose': 648, 'matellic grey': 649, 'matlic peach': 650,
         'matt silver': 651, 'meca matelic blue ': 652, 'mehndi colour': 653, 'mess rose color': 654, 'metal blu': 655,
         'metalic': 656, 'metalic gray': 657, 'metalic green': 658, 'metalic grey': 659, 'metalic s': 660,
         'metalic steel grey': 661, 'metallic': 662, 'metallic blue': 663, 'metallic grey': 664,
         'metallic light blue': 665, 'metallic pink': 666, 'metallic purple': 667, 'metronic griy': 668,
         'mettalic green': 669, 'mettalic pink': 670, 'militry green color': 671, 'mist': 672, 'modern steel': 673,
         'moigia color': 674, 'motelec darkgrey': 675, 'mtrilc pink': 676, 'multi': 677, 'navy': 678,
         'neavymatalic': 679, 'nem blue': 680, 'nnn': 681, 'not defined': 682, 'not know color name': 683,
         'oilve green': 684, 'olive Green': 685, 'olive gold': 686, 'olive green': 687, 'outer shawer iner jenean': 688,
         'p green': 689, 'p white': 690, 'pal white': 691, 'pall whith': 692, 'palwhith': 693,
         'panthon brown colour': 694, 'parl whit': 695, 'peach': 696, 'peach color': 697, 'pearl White': 698,
         'pearl White ': 699, 'pearl black': 700, 'pearl gold': 701, 'pearl grey': 702, 'pearl parpal': 703,
         'pearl sky Blue': 704, 'pearl white': 705, 'pearl white and black': 706, 'pearlwhite': 707, 'peel brown': 708,
         'peli white': 709, 'penthara metal': 710, 'penthra metal': 711, 'peo blue': 712, 'peral white': 713,
         'perelwhite': 714, 'pethera metallic': 715, 'phanthon brown colour': 716, 'pink metalic': 717,
         'pink rose': 718, 'pinkish': 719, 'pista': 720, 'pista shade greeny': 721, 'polish matellic': 722,
         'powder pink light matelic': 723, 'preal white': 724, 'prl white': 725, 'purplish blue': 726, 'read wain': 727,
         'red vine': 728, 'red wine': 729, 'redvine': 730, 'redwine': 731, 'rose': 732, 'rose Metallic': 733,
         'rose Pink': 734, 'rose color': 735, 'rose gold': 736, 'rose matalic': 737, 'rose matel': 738,
         'rose matelic': 739, 'rose metal': 740, 'rose metalic': 741, 'rose metalic(falsa color)': 742,
         'rose metallic': 743, 'rose mint like silver': 744, 'rose mist': 745, 'rose mitalic': 746, 'rose pink': 747,
         'rose silver': 748, 'rosemetilk': 749, 'rough': 750, 'royal blu e': 751, 's green': 752, 's rose': 753,
         'saloon': 754, 'saloon gray': 755, 'sand bedige': 756, 'sand beig': 757, 'sand beige': 758, 'sand gold': 759,
         'sea gerrn': 760, 'sea green': 761, 'sedan': 762, 'see dark green': 763, 'shaleemar rose': 764,
         'shalimar': 765, 'shalimar rose': 766, 'shalimar stone': 767, 'sheamring green color': 768,
         'shemreen green': 769, 'shemring green': 770, 'shimer bist': 771, 'shimmeer green': 772,
         'shimmering Green': 773, 'shimmering grean': 774, 'shimmering green': 775, 'shimmering green ': 776,
         'shimran green': 777, 'shine Golden': 778, 'shining light blue': 779, 'shmiran green': 780,
         'shmrin green': 781, 'silky green': 782, 'silve & Green': 783, 'silver Green': 784, 'silver green': 785,
         'silver metallic': 786, 'silverish blue': 787, 'sky': 788, 'sky Blue': 789, 'sky blue': 790, 'sky blue ': 791,
         'sky blue matalic': 792, 'sky blue metalic': 793, 'sky bule': 794, 'sky gray': 795, 'skyblue': 796,
         'sliver grey': 797, 'smoke GR': 798, 'smoke Green': 799, 'smoke Grey': 800, 'smoke green': 801,
         'smoke green colour': 802, 'smoke grey': 803, 'smook green': 804, 'space grey': 805, 'special edition': 806,
         'sporty orange': 807, 'spy blue': 808, 'ssss': 809, 'steel grey': 810, 'stone grey': 811, 'strong blue': 812,
         'super whit': 813, 'super white': 814, 'titanium': 815, 'tolorado blue': 816, 'tomato red': 817, 'trim': 818,
         'two color, top White': 819, 'two toned color': 820, 'unlidted': 821, 'unlisted': 822, 'urban Titanium': 823,
         'urban metallic': 824, 'urban platinum': 825, 'urban tatinum': 826, 'urban titan': 827, 'urban titanium': 828,
         'urban titanium ': 829, 'urban titanium metallic': 830, 'urban titanium.': 831, 'urban titenium': 832,
         'urban-titanium': 833, 'urbantatinam': 834, 'v. mist': 835, 'viking emerald': 836, 'vine red': 837,
         'violent': 838, 'voilet metalic': 839, 'warm silver metallic': 840, 'wheatish': 841, 'white and Golden': 842,
         'white.golden': 843, 'wine': 844, 'wine red': 845, 'wine red color': 846, 'winter grey metallic': 847,
         'yellow and black': 848, 'yellow black': 849}, inplace=True)

    df['brand'].replace(
        {'Adam': 0, 'Audi': 1, 'Austin': 2, 'BAIC': 3, 'BMW': 4, 'Bentley': 5, 'Buick': 6, 'Cadillac': 7, 'Changan': 8,
         'Chery': 9, 'Chevrolet': 10, 'Chrysler': 11, 'Citroen': 12, 'Classic Cars': 13, 'DFSK': 14, 'Daehan': 15,
         'Daewoo': 16, 'Daihatsu': 17, 'Datsun': 18, 'Dodge': 19, 'Dongfeng': 20, 'FAW': 21, 'Fiat': 22, 'Ford': 23,
         'GMC': 24, 'Geely': 25, 'Golden Dragon': 26, 'Haval': 27, 'Hino': 28, 'Honda': 29, 'Hummer': 30, 'Hyundai': 31,
         'Isuzu': 32, 'JAC': 33, 'JMC': 34, 'JW Forland': 35, 'Jeep': 36, 'KIA': 37, 'Land Rover': 38, 'Lexus': 39,
         'MG': 40, 'MINI': 41, 'Master': 42, 'Mazda': 43, 'Mercedes Benz': 44, 'Mitsubishi': 45, 'Morris': 46,
         'Nissan': 47, 'Others': 48, 'Peugeot': 49, 'Plymouth': 50, 'Pontiac': 51, 'Porsche': 52, 'Prince': 53,
         'Proton': 54, 'Range Rover': 55, 'Renault': 56, 'Roma': 57, 'Saab': 58, 'Skoda': 59, 'Sogo': 60, 'Sokon': 61,
         'SsangYong': 62, 'Subaru': 63, 'Suzuki': 64, 'Tesla': 65, 'Toyota': 66, 'United': 67, 'Vauxhall': 68,
         'Volkswagen': 69, 'Volvo': 70, 'Willys': 71, 'ZOTYE': 72}, inplace=True)

    df['model'].replace(
        {'1 Series': 0, '1000': 1, '120 Y': 2, '1200': 3, '1300': 4, '200 D': 5, '200 T': 6, '2008': 7, '205': 8,
         '240 Gd': 9, '250 D': 10, '3': 11, '3 Series': 12, '300 C': 13, '300 Series': 14, '323': 15, '350Z': 16,
         '370Z': 17, '5': 18, '5 Series': 19, '6 Series': 20, '626': 21, '7 Series': 22, '86': 23, '929': 24,
         'A Class': 25, 'A1': 26, 'A3': 27, 'A4': 28, 'A5': 29, 'A6': 30, 'A800': 31, 'AD Van': 32, 'APV': 33,
         'Accent': 34, 'Accord': 35, 'Acty': 36, 'Acura': 37, 'Aerio': 38, 'Airwave': 39, 'Allion': 40, 'Almera': 41,
         'Alpha': 42, 'Alphard': 43, 'Alphard G': 44, 'Alphard Hybrid': 45, 'Alsvin': 46, 'Altezza': 47, 'Alto': 48,
         'Alto Lapin': 49, 'Anglia': 50, 'Aqua': 51, 'Atenza Wagon': 52, 'Atrai Wagon': 53, 'Auris': 54,
         'Autobiography': 55, 'Avanza': 56, 'Avensis': 57, 'Aveo': 58, 'Axela': 59, 'Aygo': 60, 'Azwagon': 61,
         'Azwagon Custom Style': 62, 'B2200': 63, 'BJ40': 64, 'BJ40 Plus': 65, 'BR-V': 66, 'Baleno': 67, 'Beat': 68,
         'Beetle': 69, 'Bego': 70, 'Belta': 71, 'Besturn': 72, 'Bj212': 73, 'Blue Bird': 74, 'Bluebird Sylphy': 75,
         'Bolan': 76, 'Boltoro': 77, 'Bongo': 78, 'Boon': 79, 'Brabus ': 80, 'Bravo': 81, 'C Class': 82,
         'C Class Coupe': 83, 'C Class Estate': 84, 'C-10 ': 85, 'C-19': 86, 'C-314': 87, 'C-717': 88, 'C-HR': 89,
         'C37': 90, 'CJ 5': 91, 'CLA Class': 92, 'CLK Class': 93, 'CLS Class': 94, 'CR-V': 95, 'CR-Z Sports Hybrid': 96,
         'CT200h': 97, 'CX70T': 98, 'Caldina': 99, 'Cami': 100, 'Camry': 101, 'Caprice': 102, 'Caravan': 103,
         'Carina': 104, 'Carisma': 105, 'Carol': 106, 'Carol Eco': 107, 'Carrier': 108, 'Carry': 109, 'Cast': 110,
         'Cayenne': 111, 'Cedric': 112, 'Cefiro': 113, 'Celerio': 114, 'Celica': 115, 'Cerato': 116, 'Cervo': 117,
         'Challenger': 118, 'Charade': 119, 'Charmant': 120, 'Chaser': 121, 'Cherokee': 122, 'Cherry': 123,
         'Chitral': 124, 'Ciaz': 125, 'Cielo': 126, 'City': 127, 'Civic': 128, 'Civic Hybrid': 129, 'Cj 6': 130,
         'Cj 7': 131, 'Ck': 132, 'Cl Class': 133, 'Classic': 134, 'Clipper': 135, 'Coaster': 136,
         'Commander 5.7 V8 Hemi': 137, 'Concerto': 138, 'Continental Gt': 139, 'Convoy': 140, 'Coo': 141, 'Cooper': 142,
         'Copen': 143, 'Corolla': 144, 'Corolla Assista': 145, 'Corolla Axio': 146, 'Corolla Cross': 147,
         'Corolla Fielder': 148, 'Corona': 149, 'Cortina': 150, 'Corvette': 151, 'Coupe': 152, 'Cressida': 153,
         'Cresta': 154, 'Cross Road': 155, 'Crown': 156, 'Cruze': 157, 'Cts': 158, 'Cube': 159, 'Cultus': 160,
         'Cuore': 161, 'Cx3': 162, 'D Series': 163, 'D-Max': 164, 'Dayz': 165, 'Dayz Highway Star': 166,
         'Defender': 167, 'Demio': 168, 'Dias Wagon': 169, 'Discovery': 170, 'Domingo': 171, 'Double Cabin': 172,
         'Duet': 173, 'Duster': 174, 'E 2200': 175, 'E Class': 176, 'E Class Cabriolet': 177, 'E Class Coupe': 178,
         'EK Custom': 179, 'EK Space Custom': 180, 'Ek Sport': 181, 'Ek Wagon': 182, 'Elantra': 183, 'Element': 184,
         'Escalade Ext': 185, 'Esquire': 186, 'Esse': 187, 'Estima': 188, 'Every': 189, 'Every Wagon': 190,
         'Evoque': 191, 'Excel': 192, 'Exclusive': 193, 'F 150': 194, 'F 150 Shelby': 195, 'FUSO': 196, 'FX': 197,
         'Familia Van': 198, 'Family Van': 199, 'Ferio': 200, 'Feroza': 201, 'Fiesta': 202, 'Fit': 203, 'Fit Aria': 204,
         'Fj Cruiser': 205, 'Flair': 206, 'Flair Custom Style': 207, 'Flair Wagon': 208, 'Forester': 209,
         'Fortuner': 210, 'Foton': 211, 'Freed': 212, 'Freelander': 213, 'Freelander 2': 214, 'Fto': 215, 'Fx4': 216,
         'G Class': 217, 'GLA Class': 218, 'Galant': 219, 'Gen 2': 220, 'Gilgit': 221, 'Glory 500': 222,
         'Glory 580': 223, 'Gn250': 224, 'Golf': 225, 'Grace': 226, 'Grace Hybrid': 227, 'Grand Carnival': 228,
         'Grand Starex': 229, 'Gs Series': 230, 'Gto': 231, 'Guru': 232, 'H-100': 233, 'H1': 234, 'H2': 235, 'H6': 236,
         'HR-V': 237, 'HS': 238, 'Harrier': 239, 'Hiace': 240, 'Hijet': 241, 'Hilux': 242, 'Hse 4.6': 243,
         'Hustler': 244, 'I': 245, 'I Mivec': 246, 'ISIS': 247, 'IST': 248, 'Ignis': 249, 'Impreza': 250,
         'Impreza Sports': 251, 'Insight': 252, 'Insight Exclusive': 253, 'Inspire': 254, 'Integra': 255, 'Jade': 256,
         'Jiaxing Mini Mpvs': 257, 'Jimny': 258, 'Jimny Sierra': 259, 'Jolion': 260, 'Joy': 261, 'Juke': 262,
         'Justy': 263, 'K01': 264, 'K07': 265, 'Kaghan XL': 266, 'Kalam': 267, 'Kalash': 268, 'Karvaan': 269,
         'Kei': 270, 'Khyber': 271, 'Kicks': 272, 'Kix': 273, 'Kizashi': 274, 'Korando': 275, 'L200': 276, 'L300': 277,
         'LX Series': 278, 'Lancer': 279, 'Lancer Evolution': 280, 'Land Cruiser': 281, 'Latio': 282, 'Le Mans': 283,
         'Liana': 284, 'Life': 285, 'Lite Ace': 286, 'Lj80': 287, 'Luce': 288, 'Lucida': 289, 'Lucra': 290,
         'M 151': 291, 'M 825': 292, 'M Class': 293, 'M Series': 294, 'M38': 295, 'M8': 296, 'M9': 297, 'MR Wagon': 298,
         'MR2': 299, 'Macan': 300, 'March': 301, 'Margalla': 302, 'Mark II': 303, 'Mark X': 304, 'Matiz': 305,
         'Mb140': 306, 'Mega Carry Xtra': 307, 'Mehran': 308, 'Micra': 309, 'Midget': 310, 'Mini': 311, 'Minica': 312,
         'Minicab Bravo': 313, 'Mira': 314, 'Mira Cocoa': 315, 'Mira Gino': 316, 'Mirage': 317, 'Mobilio': 318,
         'Moco': 319, 'Model S': 320, 'Move': 321, 'Move Canbus': 322, 'Move Conte': 323, 'Move Latte': 324, 'Mpv': 325,
         'Mustang': 326, 'Mutt M 825': 327, 'N Box': 328, 'N Box Custom': 329, 'N Box Plus': 330,
         'N Box Plus Custom': 331, 'N Box Slash': 332, 'N One': 333, 'N Wgn': 334, 'NKR': 335, 'Navara': 336,
         'New Boarding': 337, 'Nitro': 338, 'Noah': 339, 'Note': 340, 'Nv200 Vanette Wagon': 341, 'Nx': 342,
         'Opti': 343, 'Optra': 344, 'Other': 345, 'Otti': 346, 'Oxford': 347, 'Pajero': 348, 'Pajero Junior': 349,
         'Pajero Mini': 350, 'Palette': 351, 'Panamera': 352, 'Passo': 353, 'Path Finder': 354, 'Patrol': 355,
         'Pearl': 356, 'Picanto': 357, 'Pickup': 358, 'Pino': 359, 'Pixis Epoch': 360, 'Pixis Space': 361,
         'Pixis Van': 362, 'Pixo': 363, 'Platz': 364, 'Pleo': 365, 'Porte': 366, 'Potohar': 367, 'Prado': 368,
         'Premio': 369, 'Previa': 370, 'Pride': 371, 'Primera': 372, 'Prius': 373, 'Prius Alpha': 374, 'Probox': 375,
         'Q2': 376, 'Q3': 377, 'Q5': 378, 'Q7': 379, 'QQ': 380, 'Qashqai': 381, 'R2': 382, 'RX Series': 383, 'RX8': 384,
         'Racer': 385, 'Raize': 386, 'Ram': 387, 'Ranger': 388, 'Raum': 389, 'Rav4': 390, 'Ravi': 391, 'Revo': 392,
         'Rexton': 393, 'Rocket': 394, 'Rocky': 395, 'Roomy': 396, 'Roox': 397, 'Runx': 398, 'Rush': 399, 'Rustom': 400,
         'Rvr': 401, 'S Class': 402, 'S40': 403, 'S660': 404, 'SLK Class': 405, 'Safari': 406, 'Saga': 407,
         'Sambar ': 408, 'Samuari': 409, 'Samurai': 410, 'Santro': 411, 'Scrum': 412, 'Scrum Wagon': 413, 'Se 4.0': 414,
         'Senova X25': 415, 'Senya R7 ': 416, 'Sera': 417, 'Serena': 418, 'Shahanshah': 419, 'Shahbaz': 420,
         'Shehzore': 421, 'Shineray': 422, 'Shogun': 423, 'Sienta': 424, 'Sierra': 425, 'Silverado': 426, 'Sirion': 427,
         'Sirius': 428, 'Sj410': 429, 'Skyline': 430, 'Smart': 431, 'Smart Forfour': 432, 'Smart Fortwo': 433,
         'Solio': 434, 'Sonata': 435, 'Sonica': 436, 'Sorento': 437, 'Spacia': 438, 'Spade': 439, 'Spark': 440,
         'Spectra': 441, 'Spike': 442, 'Splash': 443, 'Sport': 444, 'Sportage': 445, 'Sprinter': 446, 'Starlet': 447,
         'Stavic': 448, 'Stella': 449, 'Stinger': 450, 'Stonic': 451, 'Storia': 452, 'Stream': 453, 'Succeed': 454,
         'Sunny': 455, 'Surf': 456, 'Swift': 457, 'Sx4': 458, 'T6': 459, 'TM 2.0': 460, 'Tacoma': 461, 'Taft': 462,
         'Tank': 463, 'Tanto': 464, 'Terios': 465, 'Terios Kid': 466, 'Terrano': 467, 'Thats': 468, 'Thor': 469,
         'Tiida': 470, 'Titan': 471, 'Toppo': 472, 'Touareg': 473, 'Town Ace': 474, 'Town Box': 475, 'Toyo Ace': 476,
         'Triton': 477, 'Trooper': 478, 'Tt': 479, 'Tucson': 480, 'Tundra': 481, 'Uno': 482, 'V2': 483, 'V40': 484,
         'Vamos': 485, 'Vamos Hobio': 486, 'Van': 487, 'Verossa': 488, 'Vezel': 489, 'Vitara': 490, 'Vito': 491,
         'Vitz': 492, 'Vogue': 493, 'Voxy': 494, 'Wagon R': 495, 'Wake': 496, 'Wingroad': 497, 'Wish': 498,
         'Wrangler': 499, 'X': 500, 'X Trail': 501, 'X-PV': 502, 'X1': 503, 'X2': 504, 'X200': 505, 'X3 Series': 506,
         'X5 Series': 507, 'X6 Series': 508, 'X70': 509, 'Xbee': 510, 'Xml6532': 511, 'Yaris': 512, 'Yaris Cross': 513,
         'Yaris Hatchback': 514, 'Z100': 515, 'ZS': 516, 'Zabardast': 517, 'Zest': 518, 'Zest Spark': 519, 'i8': 520,
         'iQ': 521, 'iX': 522}, inplace=True)

    return df
@app.route('/result',methods=['POST'])
def results():

    request_data = request.get_json(force=True)
    data = [np.array(list(request_data.values()))]
    print(data)
    header = ['brand', 'model', 'modelDate', 'fuelType', 'vehicleTransmission',
       'color', 'mileageFromOdometer', 'engDisp']
    test_df = pd.DataFrame(data=data, columns=header)
    test_df = data_encoding(test_df)
    test_arr = test_df.iloc[:, :]
    test_arr = np.asarray(test_arr).astype(np.float32)

    mean = np.array([5.1999287e+01, 2.2259625e+02, 2.0105276e+03, 1.7225736e-01, 4.8581758e-01, 3.1922900e+02, 9.2734234e+04, 1.4075039e+03])
    std = np.array([1.8115810e+01, 1.3277213e+02, 8.7116079e+00, 5.7119268e-01, 4.9979883e-01, 1.7810933e+02, 9.0697312e+04, 6.7140497e+02])

    test_arr-=mean
    test_arr /= std

    prediction = loaded_model.predict(test_arr)

    print(prediction[0])
    return prediction[0]
if __name__ == "__main__":
    app.run()

