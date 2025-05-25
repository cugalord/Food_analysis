import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set(style="whitegrid")

get_Q_name = reverse_dict = {
    'subvencionirani_obroki': 'Kako pogosto na teden koristite študentske subvencionirane obroke? ',
    'sub_obrok_hitra_hrana': 'Restavracije s hitro prehrano (Mcdonalds, Ajda, Kebab…) ',
    'sub_obrok_menza': 'Študentska menza  ',
    'sub_obrok_klasika': 'Klasične restavracije (Foculus, Parma, Kratochwill…)',
    'sub_obrok_drugo': 'Drugo:',
    'sub_obrok_drugo_text': 'Drugo:.1',
    'doma_teden': 'Kolikokrat na teden jeste doma pripravljeno hrano?',
    'ocena_prehrane': 'Kako bi ocenili svojo splošno prehrano',
    'dejavniki_cena': 'Cena',
    'dejavniki_okus': 'Okus',
    'dejavniki_zdravje': 'Hranilna vrednost / vpliv na zdravje',
    'dejavniki_prirocnost': 'Prihranek časa / priročnost',
    'dejavniki_druzba': 'Družbeni vpliv (npr. prijatelji, družina)',
    'dejavniki_lokacija': 'Dostopnost / lokacija',
    'trditve_subcena': 'Subvencionirane obroke izberem predvsem zaradi cene.',
    'trditve_domacas': 'Pogosteje bi kuhal/-a doma, če bi imel/-a več časa.  ',
    'trditve_kontrola': 'Občutek imam, da imam dovolj veliko kontrolo nad svojim prehranjevanjem.',
    'trditve_frend': 'Izbira hrane je odvisna tudi od izbire restavracije, ki jo predlagajo prijatelji.',
    'trditve_vpliv': 'Zelo me skrbi dolgoročen vpliv moje trenutne prehrane na zdravje.  ',
    'trditve_subzdravdrag': 'Zdi se mi, da je zdrava prehrana za študente predraga.  ',
    'trditve_hranilna': 'Pri izbiri obroka vedno upoštevam njegovo hranilno vrednost.  ',
    'trditve_subzdrav': 'Menim, da je tipičen študentski subvencionirani obrok zdrav.  ',
    'starost': 'Koliko ste stari? (Vpišite letnico rojstva)',
    'spol': 'Kakšen je vaš spol?',
    'bivanje': 'Kje živite?',
    'bivanje_drugo': 'Drugo:.2',
    'področje': 'Na katerem področju študirate?',
    'področje_drugo': 'Drugo:.3',
    'stopnja': 'Katero stopnjo študija trenutno obiskujete?',
    'mag_letnik': 'Kateri letnik?',
    'dip_letnik': 'Kateri letnik?.1'
}

subvencionirani_obroki = {
        "0-krat": 1,
        "1–3-krat": 2,
        "4–6-krat": 3,
        "7–9-krat": 4,
        "10-krat ali več": 5
}
subvencionirani_obroki_rev = {v: k for k, v in subvencionirani_obroki.items()}


doma_teden ={
        "0-krat": 1,
        "1–2-krat": 2,
        "3–5-krat": 3,
        "6–9-krat": 4,
        "10-krat ali več": 5,
}
doma_teden_rev = {v: k for k, v in doma_teden.items()}


ocena_prehrane = {
        "Zelo nezdrava": 1,
        "Nezdrava": 2,
        "Delno nezdrava": 3,
        "Delno zdrava": 4,
        "Zdrava": 5,
        "Zelo zdrava": 6
}
ocena_prehrane_rev = {v: k for k, v in ocena_prehrane.items()}


dejavniki = {
'Nima vpliva': 1,
'Majhen vpliv': 2,
'Velik vpliv': 3,
'Zelo velik vpliv': 4,
'Ne morem oceniti / Nimam mnenja': 5
}
dejavniki_rev = {v: k for k, v in dejavniki.items()}

trditve = {
'Popolnoma se ne strinjam': 1,
'Delno se ne strinjam': 2,
'Niti se strinjam niti ne strinjam': 3,
'Delno se strinjam': 4,
'Popolnoma se strinjam': 5,
'Ne morem oceniti / Nimam mnenja': 6
}
trditve_rev = {v: k for k, v in trditve.items()}

spol = {
'Moški': 1,
'Ženski': 2,
'Ostalo': 3,
'Ne želim odgovoriti': 4
}
spol_rev = {v: k for k, v in spol.items()}

bivanje ={
'Pri starših ali družini': 1,
'V študentskem domu': 2,
'V lastnem stanovanju': 3,
'Drugo': 4,
'Ne želim odgovoriti': 5
}
bivanje_rev = {v: k for k, v in bivanje.items()}

stopnja = {
'Dodiplomski študij': 1,
'Magistrski študij': 2,
'Doktorski študij ': 3
}
stopnja_rev = {v: k for k, v in stopnja.items()}

področje ={
'Naravoslovje, tehnika ali informatika':1,
'Družboslovje ali humanistika':2,
'Zdravstvo ali medicina':3,
'Pedagoške vede':4,
'Umetnost':5,
'Ekonomija':6,
'Pravo':7,
'Drugo':8,
'Ne želim odgovoriti':9
}
področje_rev = {v: k for k, v in področje.items()}

def load_and_clean_data(filepath):
    # Naloži CSV brez prve vrste (glava z vprašanji)
    df = pd.read_csv(filepath, sep=";", skiprows=[0], encoding="utf-8", on_bad_lines='skip')

    # Očisti morebitne ="" zapise
    df = df.apply(lambda col: col.map(lambda x: str(x).replace('="', '').replace('"', '') if isinstance(x, str) else x))
    #print(df.columns)
    # Preimenuj ustrezne stolpce
    df = df.rename(columns={
        'Kako pogosto na teden koristite študentske subvencionirane obroke? ': 'subvencionirani_obroki',
        'Restavracije s hitro prehrano (Mcdonalds, Ajda, Kebab…) ': 'sub_obrok_hitra_hrana',
        'Študentska menza  ': 'sub_obrok_menza',
        'Klasične restavracije (Foculus, Parma, Kratochwill…)': 'sub_obrok_klasika',
        'Drugo:': 'sub_obrok_drugo',
        'Drugo:.1': 'sub_obrok_drugo_text',
        'Kolikokrat na teden jeste doma pripravljeno hrano?': 'doma_teden',
        'Kako bi ocenili svojo splošno prehrano?': 'ocena_prehrane',
        'Cena':'dejavniki_cena',
        'Okus':'dejavniki_okus',
        'Hranilna vrednost / vpliv na zdravje':'dejavniki_zdravje',
        'Prihranek časa / priročnost':'dejavniki_prirocnost',
        'Družbeni vpliv (npr. prijatelji, družina)':'dejavniki_druzba',
        'Dostopnost / lokacija':'dejavniki_lokacija',
        'Subvencionirane obroke izberem predvsem zaradi cene.': 'trditve_subcena',
        'Pogosteje bi kuhal/-a doma, če bi imel/-a več časa.  ': 'trditve_domacas',
        'Občutek imam, da imam dovolj veliko kontrolo nad svojim prehranjevanjem.': 'trditve_kontrola',
        'Izbira hrane je odvisna tudi od izbire restavracije, ki jo predlagajo prijatelji.': 'trditve_frend',
        'Zelo me skrbi dolgoročen vpliv moje trenutne prehrane na zdravje.  ': 'trditve_vpliv',
        'Zdi se mi, da je zdrava prehrana za študente predraga.  ': 'trditve_subzdravdrag',
        'Pri izbiri obroka vedno upoštevam njegovo hranilno vrednost.  ': 'trditve_hranilna',
        'Menim, da je tipičen študentski subvencionirani obrok zdrav.  ': 'trditve_subzdrav',
        'Koliko ste stari? (Vpišite letnico rojstva)': 'starost',
        'Kakšen je vaš spol?':'spol',
        'Kje živite?':'bivanje',
        'Drugo:.2':'bivanje_drugo',
        'Na katerem področju študirate?':'področje',
        'Drugo:.3':'področje_drugo',
        'Katero stopnjo študija trenutno obiskujete?':'stopnja',
        'Kateri letnik?':'mag_letnik',
        'Kateri letnik?.1':'dip_letnik',
    })

    df["subvencionirani_obroki"] = df["subvencionirani_obroki"].astype(int)
    df["sub_obrok_hitra_hrana"] = df["sub_obrok_hitra_hrana"].astype(int)
    df["sub_obrok_menza"] = df["sub_obrok_menza"].astype(int)
    df["sub_obrok_klasika"] = df["sub_obrok_klasika"].astype(int)
    df["sub_obrok_drugo"] = df["sub_obrok_drugo"].astype(int)
    df["ocena_prehrane"] = df["ocena_prehrane"].astype(int)
    df['dejavniki_cena']=df['dejavniki_cena'].astype(int)
    df['dejavniki_okus']=df['dejavniki_okus'].astype(int)
    df['dejavniki_zdravje']=df['dejavniki_zdravje'].astype(int)
    df['dejavniki_prirocnost']=df['dejavniki_prirocnost'].astype(int)
    df['dejavniki_druzba']=df['dejavniki_druzba'].astype(int)
    df['dejavniki_lokacija']=df['dejavniki_lokacija'].astype(int)
    df['trditve_subcena']=df['trditve_subcena'].astype(int)
    df['trditve_domacas']=df['trditve_domacas'].astype(int)
    df['trditve_kontrola']=df['trditve_kontrola'].astype(int)
    df['trditve_frend']=df['trditve_frend'].astype(int)
    df['trditve_vpliv']=df['trditve_vpliv'].astype(int)
    df['trditve_subzdravdrag']=df['trditve_subzdravdrag'].astype(int)
    df['trditve_hranilna']=df['trditve_hranilna'].astype(int)
    df['trditve_subzdrav']=df['trditve_subzdrav'].astype(int)
    df['spol'] = df['spol'].astype(int)
    df['bivanje'] = df['bivanje'].astype(int)
    df['stopnja'] = df['stopnja'].astype(int)


    # če je vrednost negativna pomeni da na odgovor niso odgovorili zato ga spremenimo v NaN
    for col in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Replace negative values with NaN
            df[col] = df[col].apply(lambda x: np.nan if pd.notnull(x) and x < 0 else x)


    # age ajdustment
    df['starost'] = pd.to_numeric(df['starost'], errors='coerce')

    df['starost'] = df['starost'].apply(
        lambda x: 2025 - x if x > 1900 else x
    )
    df.loc[df['starost'] < 18, 'starost'] = np.nan
    df.loc[df['starost'] > 35, 'starost'] = np.nan
    df['starost'] = df['starost'].astype('Int64')

    # Odstrani manjkajoče
    df_clean = df.dropna(subset=["subvencionirani_obroki", "ocena_prehrane", "doma_teden","sub_obrok_hitra_hrana","sub_obrok_menza","sub_obrok_klasika", "sub_obrok_drugo", "sub_obrok_drugo_text", 'dejavniki_cena',
    'dejavniki_okus',
    'dejavniki_zdravje',
    'dejavniki_prirocnost',
    'dejavniki_druzba',
    'dejavniki_lokacija',
    'trditve_subcena',
    'trditve_domacas',
    'trditve_kontrola',
    'trditve_frend',
    'trditve_vpliv',
    'trditve_subzdravdrag',
    'trditve_hranilna',
    'trditve_subzdrav',
    'starost',
    'spol',
    'bivanje',
    'stopnja'
    ])
    return df_clean

def subvencionirani_obroki(df):
    """Izpiši osnovne statistike."""
    print('Vprašanje: ' + get_Q_name["subvencionirani_obroki"])

    print("\n📊 Deskriptivna statistika:")
    pd.options.display.float_format = '{:.2f}'.format
    print(df[["subvencionirani_obroki"]].describe())

    if 'subvencionirani_obroki' in df.columns:
        # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
        temp_subvencionirani_obroki_numeric = pd.to_numeric(df["subvencionirani_obroki"], errors='coerce').dropna()

        # Preslikaj številčne vrednosti v opisne za prikaz
       # print(subvencionirani_obroki_rev)
        df_display_hist = temp_subvencionirani_obroki_numeric.map(subvencionirani_obroki_rev)
        plt.figure(figsize=(8, 6))

        # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
        # določite vrstni red z uporabo ključev iz reverse_ocena_map
        order = [subvencionirani_obroki_rev[i] for i in (subvencionirani_obroki_rev.keys())]

        print(df_display_hist)
        sns.countplot(df_display_hist, order=order, stat='count')

        plt.title(get_Q_name['subvencionirani_obroki'])
        plt.xlabel("Število študentov")
        plt.ylabel("")

        # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

def sub_obrok_tip(df):
    def is_true_string(x):
        if not isinstance(x, str):
            return False
        try:
            float(x)
            return False  # Numeric string
        except ValueError:
            return True  # Truly a non-numeric string
    print('Vprašanje: Kje običajno jeste subvencionirane obroke? (Označite vse, kar velja)')

    # Primer: samo stolpci z 0/1 vrednostmi, npr. sub_obrok stolpci
    columns = ['sub_obrok_hitra_hrana', 'sub_obrok_menza', 'sub_obrok_klasika', 'sub_obrok_drugo']
    # Seštejemo 1-ke po stolpcih
    counts = [df[column].sum() for column in columns]
    columns=[get_Q_name[column] for column in columns]
    print("DRUGO: ")
    drugo_opcije=df['sub_obrok_drugo_text']
    for x in drugo_opcije:
        if is_true_string(x):
            print('\t'+ x+',')

    # Ustvarimo stolpčni graf
    plt.figure(figsize=(10, 12))
    sns.barplot(x=columns, y=counts)
    plt.ylabel('Število')
    plt.xlabel('')
    plt.title('Kolikokrat je bila posamezna izbira označena')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def doma_teden(df):
    """Izpiši osnovne statistike."""
    print('Vprašanje: ' + get_Q_name["doma_teden"])

    print("\n📊 Deskriptivna statistika:")
    pd.options.display.float_format = '{:.2f}'.format
    print(df["doma_teden"].describe())

    if 'doma_teden' in df.columns:
        # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
        temp_doma_teden_numeric = pd.to_numeric(df["doma_teden"], errors='coerce').dropna()

        # Preslikaj številčne vrednosti v opisne za prikaz
       # print(doma_teden_rev)
        df_display_hist = temp_doma_teden_numeric.map(doma_teden_rev)
        plt.figure(figsize=(8, 6))

        # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
        # določite vrstni red z uporabo ključev iz reverse_ocena_map
        order = [doma_teden_rev[i] for i in (doma_teden_rev.keys())]

        sns.countplot(df_display_hist, order=order, stat='count')

        plt.title(get_Q_name['doma_teden'])
        plt.xlabel("Število študentov")
        plt.ylabel("")

        # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

def ocena_prehrane(df):
    print('Vprašanje: ' + get_Q_name["ocena_prehrane"])

    print("\n📊 Deskriptivna statistika:")
    pd.options.display.float_format = '{:.2f}'.format
    print(df["ocena_prehrane"].describe())

    if 'ocena_prehrane' in df.columns:
        # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
        temp_ocena_prehrane_numeric = pd.to_numeric(df["ocena_prehrane"], errors='coerce').dropna()

        # Preslikaj številčne vrednosti v opisne za prikaz
        df_display_hist = temp_ocena_prehrane_numeric.map(ocena_prehrane_rev)

        plt.figure(figsize=(8, 6))

        # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
        # določite vrstni red z uporabo ključev iz reverse_ocena_map
        order = [ocena_prehrane_rev[i] for i in sorted(ocena_prehrane_rev.keys())]

        sns.countplot(df_display_hist, order=order, stat='count')

        plt.title("Distribucija ocen prehrane")
        plt.xlabel("Število študentov")  # Spremenjeno v "Ocena prehrane", saj so etikete že opisne
        plt.ylabel("Ocena prehrane")

        # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

def loop_dejavniki(df, dejavnik):
    pd.options.display.float_format = '{:.2f}'.format
    print(df[dejavnik].describe())

    # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
    temp_d_numeric = pd.to_numeric(df[dejavnik], errors='coerce').dropna()

    # Preslikaj številčne vrednosti v opisne za prikaz
    df_display_hist = temp_d_numeric.map(dejavniki_rev)
    plt.figure(figsize=(8, 6))

    # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
    # določite vrstni red z uporabo ključev iz reverse_ocena_map
    order = [dejavniki_rev[i] for i in (dejavniki_rev.keys())]

    sns.countplot(df_display_hist, order=order, stat='count')

    plt.title(get_Q_name[dejavnik])
    plt.xlabel("Število študentov")
    plt.ylabel("")

    # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def dejavniki(df):
    vsi_dejavniki=['dejavniki_cena','dejavniki_okus','dejavniki_zdravje','dejavniki_prirocnost','dejavniki_druzba','dejavniki_lokacija']
    print('Vprašanje: V kolikšni meri vplivajo naslednji dejavniki na vašo izbiro hrane?')
    for dejavnik in vsi_dejavniki:
        loop_dejavniki(df, dejavnik)

def loop_trditve(df, trditev):
    pd.options.display.float_format = '{:.2f}'.format
    print(df[trditev].describe())

    # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
    temp_d_numeric = pd.to_numeric(df[trditev], errors='coerce').dropna()

    # Preslikaj številčne vrednosti v opisne za prikaz
    df_display_hist = temp_d_numeric.map(trditve_rev)
    plt.figure(figsize=(8, 6))

    # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
    # določite vrstni red z uporabo ključev iz reverse_ocena_map
    order = [trditve_rev[i] for i in (trditve_rev.keys())]

    sns.countplot(df_display_hist, order=order, stat='count')

    plt.title(get_Q_name[trditev])
    plt.xlabel("Število študentov")
    plt.ylabel("")

    # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def trditve(df):
    vse_trditve=['trditve_subcena','trditve_domacas','trditve_kontrola', 'trditve_frend', 'trditve_vpliv', 'trditve_subzdravdrag', 'trditve_hranilna', 'trditve_subzdrav']
    for trditev in vse_trditve:
        loop_trditve(df, trditev)

def startost(df):
    """Izpiši osnovne statistike."""
    print('Vprašanje: ' + get_Q_name["starost"])
    print("\n📊 Deskriptivna statistika:")
    pd.options.display.float_format = '{:.2f}'.format
    print(df["starost"].describe())
    
    # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
    temp_d_numeric = pd.to_numeric(df["starost"], errors='coerce').dropna()
    
    # Preslikaj številčne vrednosti v opisne za prikaz
    df_display_hist = df['starost']

    
    # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
    # določite vrstni red z uporabo ključev iz reverse_ocena_map
    order = []
    
    df['starost'].value_counts().sort_index().plot(kind='bar', figsize=(10,5))
    plt.title(get_Q_name['starost'])
    plt.xlabel("Starost")
    plt.ylabel("Število študentov")
    
    # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def spol(df):
    """Izpiši osnovne statistike."""
    print('Vprašanje: ' + get_Q_name["spol"])

    # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
    temp_d_numeric = pd.to_numeric(df["spol"], errors='coerce').dropna()

    # Preslikaj številčne vrednosti v opisne za prikaz
    df_display_hist = temp_d_numeric.map(spol_rev)
    plt.figure(figsize=(8, 6))

    # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
    # določite vrstni red z uporabo ključev iz reverse_ocena_map
    order = [spol_rev[i] for i in (spol_rev.keys())]

    sns.countplot(df_display_hist, order=order, stat='count')

    plt.title(get_Q_name['spol'])
    plt.xlabel("Število študentov")
    plt.ylabel("Spol")

    # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def bivanje(df):
    """Izpiši osnovne statistike."""
    print('Vprašanje: ' + get_Q_name["bivanje"])

    print("\n📊 Deskriptivna statistika:")
    pd.options.display.float_format = '{:.2f}'.format
    print(df["bivanje"].describe())

    # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
    temp_d_numeric = pd.to_numeric(df["bivanje"], errors='coerce').dropna()

    # Preslikaj številčne vrednosti v opisne za prikaz

    df_display_hist = temp_d_numeric.map(bivanje_rev)
    plt.figure(figsize=(8, 6))

    # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
    # določite vrstni red z uporabo ključev iz reverse_ocena_map
    order = [bivanje_rev[i] for i in (bivanje_rev.keys())]

    sns.countplot(df_display_hist, order=order, stat='count')

    plt.title(get_Q_name['bivanje'])
    plt.xlabel("Število študentov")
    plt.ylabel("")

    # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    
def stopnja(df):
    """Izpiši osnovne statistike."""
    print('Vprašanje: ' + get_Q_name["stopnja"])

    print("\n📊 Deskriptivna statistika:")
    pd.options.display.float_format = '{:.2f}'.format
    print(df["stopnja"].describe())

    # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
    temp_d_numeric = pd.to_numeric(df["stopnja"], errors='coerce').dropna()

    # Preslikaj številčne vrednosti v opisne za prikaz

    df_display_hist = temp_d_numeric.map(stopnja_rev)
    plt.figure(figsize=(8, 6))

    # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
    # določite vrstni red z uporabo ključev iz reverse_ocena_map
    order = [stopnja_rev[i] for i in (stopnja_rev.keys())]

    sns.countplot(df_display_hist, order=order, stat='count')

    plt.title(get_Q_name['stopnja'])
    plt.xlabel("Število študentov")
    plt.ylabel("")

    # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def področje(df):
    """Izpiši osnovne statistike."""
    print('Vprašanje: ' + get_Q_name["področje"])

    print("\n📊 Deskriptivna statistika:")
    pd.options.display.float_format = '{:.2f}'.format
    print(df["področje"].describe())

    # Preverite, ali so vrednosti v stolpcu številke, preden jih mapirate.
    temp_d_numeric = pd.to_numeric(df["področje"], errors='coerce').dropna()

    # Preslikaj številčne vrednosti v opisne za prikaz

    df_display_hist = temp_d_numeric.map(področje_rev)
    plt.figure(figsize=(8, 6))

    # Uporabite pd.Categorical za zagotovitev pravilnega vrstnega reda na X-osi
    # določite vrstni red z uporabo ključev iz reverse_ocena_map
    order = [področje_rev[i] for i in (področje_rev.keys())]

    sns.countplot(df_display_hist, order=order, stat='count')

    plt.title(get_Q_name['področje'])
    plt.xlabel("Število študentov")
    plt.ylabel("")

    # Ročno nastavite X-tick etikete, da so pravilno razporejene in rotirane, če je potrebno
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def display_df(df):
    """
    Izpiše celoten Pandas DataFrame brez skrajševanja.
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\n--- Celoten DataFrame ---")
        print(df)
        print("-------------------------\n")


def korelacijska_matrika(df):
    """
    Korelacijska matrika (Pearson) za dejavnike in trditve.
    """
    kor_cols = [
    'subvencionirani_obroki',
    'sub_obrok_hitra_hrana',
    'sub_obrok_menza',
    'sub_obrok_klasika',
    'sub_obrok_drugo',
    'doma_teden',
    'ocena_prehrane',
    'dejavniki_cena',
    'dejavniki_okus',
    'dejavniki_zdravje',
    'dejavniki_prirocnost',
    'dejavniki_druzba',
    'dejavniki_lokacija',
    'trditve_subcena',
    'trditve_domacas',
    'trditve_kontrola',
    'trditve_frend',
    'trditve_vpliv',
    'trditve_subzdravdrag',
    'trditve_hranilna',
    'trditve_subzdrav',
    'starost',
    'spol',
    'bivanje',
    'področje',
    'stopnja'
    ]

    corr = df[kor_cols].corr(method='pearson')

    plt.figure(figsize=(24, 20))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=True, yticklabels=True)
    plt.title("Korelacijska matrika (Pearson)")
    plt.tight_layout()
    plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
importances_plot = True

def random_forest(df, target):
    """
    Random Forest: napovedovanje razreda 'subvencionirani_obroki' na podlagi vseh dejavnikov in trditev.
    """
    features = [
    'subvencionirani_obroki',
    'sub_obrok_hitra_hrana',
    'sub_obrok_menza',
    'sub_obrok_klasika',
    'sub_obrok_drugo',
    'doma_teden',
    'ocena_prehrane',
    'dejavniki_cena',
    'dejavniki_okus',
    'dejavniki_zdravje',
    'dejavniki_prirocnost',
    'dejavniki_druzba',
    'dejavniki_lokacija',
    'trditve_subcena',
    'trditve_domacas',
    'trditve_kontrola',
    'trditve_frend',
    'trditve_vpliv',
    'trditve_subzdravdrag',
    'trditve_hranilna',
    'trditve_subzdrav',
    'starost',
    'spol',
    'bivanje',
    'področje',
    'stopnja'
    ]
    if target not in features:
        return -2

    features.remove(target)
    df_sub = df[features + [target]].dropna()

    X = df_sub[features]
    y = df_sub[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Prikaz top 5 pomembnosti
    importances = rf.feature_importances_
    indices = importances.argsort()[::-1][:5]  # Top 5

    # Pripravi lepa imena spremenljivk
    lepa_imena = [get_Q_name.get(features[i], features[i]) for i in indices]

    plt.figure(figsize=(16, 6))
    sns.barplot(x=importances[indices], y=lepa_imena, palette='viridis')
    plt.title(get_Q_name.get(target, target))
    plt.xlabel("Relativna pomembnost")
    plt.ylabel("Spremenljivka")
    plt.tight_layout()
    plt.show()


def main():
    # Naloži in očisti podatke
    filepath = "anketa.csv"
    df = load_and_clean_data(filepath)
    #display_df(df)
    #Q1
    subvencionirani_obroki(df)
    #Q2
    sub_obrok_tip(df)
    #Q3
    doma_teden(df)
    #Q4
    ocena_prehrane(df)
    #Q5
    dejavniki(df)
    #Q6
    trditve(df)
    #Q7
    startost(df)
    #Q8
    spol(df)
    #Q9
    bivanje(df)
    #Q10
    področje(df)
    #Q11
    stopnja(df)

    random_forest(df, 'subvencioniarni_obroki')
    random_forest(df, 'ocena_prehrane')



if __name__ == "__main__":
   #main()
    filepath = "anketa.csv"
    df = load_and_clean_data(filepath)
    korelacijska_matrika(df)
