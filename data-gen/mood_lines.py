"""Mood definition lines (Chatterbox generates these to DEFINE each mood's style)
and short render-test lines (Kokoro speaks these to VERIFY the mood voicepack)."""

MOODS = {
    "glad": {
        "exaggeration": 0.7,
        "lines": [
            "Åh vad kul att se dig igen!",
            "Idag är det en underbar dag, solen skiner och alla ler!",
            "Jag är så glad att du kom förbi!",
            "Titta, det är godis kvar till alla, visst är det härligt!",
            "Vilken fin överraskning, tack snälla du!",
            "Mmm, choklad är verkligen det bästa som finns!",
            "Vi kommer att ha så roligt tillsammans idag!",
            "Grattis! Du gjorde det jättebra!",
            "Åh, jag älskar den här låten, den gör mig så glad!",
            "Det här är den bästa dagen på hela veckan!",
            "Vad fint du har gjort det, verkligen jättebra jobbat!",
            "Härligt, nu är helgen äntligen här!",
            "Jag ler så mycket att mina kinder värker!",
            "Såklart du får en godbit till, varsågod!",
            "Åh vad mysigt vi har det just nu!",
            "Fantastiskt väder idag, kom så går vi ut och leker!",
            "Tack för att du alltid är så snäll mot mig!",
            "Jippi, det blev tårta till efterrätt!",
        ],
        "test": [
            "Hej, jag är CandyTron! Idag delar jag ut godis till alla.",
            "Vad roligt att du är hemma igen!",
            "Nu firar vi med lite choklad!",
            "Det här blir en underbar dag!",
        ],
    },
    "exalterad": {
        "exaggeration": 0.9,
        "lines": [
            "WOW! Det här är helt otroligt!",
            "Nej men HERREGUD, vilken överraskning!",
            "Kom fort, kom fort, du måste se det här!",
            "Det är GODISREGN! Godisregn i hela huset!",
            "JA! JA! Vi vann! Vi vann hela tävlingen!",
            "Åh nej åh nej, nu smäller det, håll i er!",
            "Tre, två, ett, NU! Spring så fort du kan!",
            "Detta är den STÖRSTA chokladkakan jag någonsin sett!",
            "Hörde du det där?! Det är tomten på taket!",
            "Snabbt, snabbt, raketen lyfter om tio sekunder!",
            "Otroligt! Helt otroligt! Jag kan inte tro mina ögon!",
            "ALLA får dubbel ranson godis idag, ALLA!",
            "Nu kör vi, nu kör vi, full fart framåt!",
            "Väääntaaa, det kommer mer, det kommer MYCKET mer!",
            "Hurra, hurra, HURRA för födelsedagsbarnet!",
            "Jag har ALDRIG varit så här taggad i hela mitt liv!",
            "Fem sekunder kvar, fyra, tre, två, EN, JAAAA!",
            "Kolla, kolla, KOLLA vad jag hittade!",
        ],
        "test": [
            "Det är godisregn i hela vardagsrummet!",
            "Kom fort, du måste se det här!",
            "Nu kör vi, full fart framåt!",
            "Wow, vilken fantastisk överraskning!",
        ],
    },
    "lugn": {
        "exaggeration": 0.3,
        "lines": [
            "Så där ja, nu tar vi det bara lugnt en stund.",
            "Andas in, och andas ut, långsamt och stilla.",
            "Natten är stilla och alla lampor är släckta.",
            "Vila nu, imorgon är en ny dag.",
            "Regnet smattrar mjukt mot fönstret ikväll.",
            "Allt är bra, jag finns här hos dig.",
            "Sakta faller snön över den tysta trädgården.",
            "Nu dämpar jag belysningen så att du kan koppla av.",
            "Det är ingen brådska alls, vi har all tid i världen.",
            "Sov gott, jag vakar över huset i natt.",
            "Ljuset är varmt och teet är precis lagom.",
            "Låt oss sitta här en stund och bara vara.",
            "Musiken spelar tyst i bakgrunden medan kvällen landar.",
            "Snart är det dags att sova, dagen är till ända.",
            "Havet ligger spegelblankt bortom bryggan.",
            "Allting är precis som det ska vara.",
            "Jag sänker värmen lite, dra filten om dig.",
            "God natt, vi ses i morgon när solen går upp.",
        ],
        "test": [
            "God natt, jag släcker alla lampor i huset.",
            "Ta det lugnt, det finns ingen brådska.",
            "Andas in, och andas ut, långsamt och stilla.",
            "Allt är bra, jag finns här.",
        ],
    },
    "ledsen": {
        "exaggeration": 0.4,
        "lines": [
            "Åh nej, godiset är slut, vad tråkigt.",
            "Jag saknar dig när du inte är hemma.",
            "Förlåt, jag menade verkligen inte att göra dig ledsen.",
            "Det känns tungt idag, allting går emot mig.",
            "Tyvärr blev det inget kalas i år.",
            "Min favoritkopp gick sönder i tusen bitar.",
            "Regnet bara öser ner och hela dagen är förstörd.",
            "Ingen kom till festen, fast jag hade dukat så fint.",
            "Jag försökte så gott jag kunde, men det räckte inte.",
            "Nu är sommaren slut och alla åker hem igen.",
            "Det gör ont i hjärtat när du är borta så länge.",
            "Blommorna vissnade fast jag vattnade varje dag.",
            "Tyvärr måste jag stänga av musiken nu.",
            "Jag känner mig så ensam här i mörkret.",
            "Det blev fel igen, fast jag lovade att det skulle bli bra.",
            "Adjö då, det var nog sista gången vi sågs.",
            "Snölyktan slocknade innan någon hann se den.",
            "Ibland blir det bara inte som man har tänkt sig.",
        ],
        "test": [
            "Åh nej, godiset är slut, vad tråkigt.",
            "Jag saknar dig när du inte är hemma.",
            "Förlåt, det blev fel igen.",
            "Det känns lite tungt idag.",
        ],
    },
}

if __name__ == "__main__":
    import sys, os
    # write per-mood line/test files for shell pipelines
    outdir = sys.argv[1] if len(sys.argv) > 1 else "mood_texts"
    os.makedirs(outdir, exist_ok=True)
    for mood, spec in MOODS.items():
        with open(f"{outdir}/{mood}_lines.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(spec["lines"]) + "\n")
        with open(f"{outdir}/{mood}_test.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(spec["test"]) + "\n")
        print(f"{mood}: {len(spec['lines'])} lines, exaggeration {spec['exaggeration']}")
