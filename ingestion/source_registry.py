from dataclasses import dataclass


@dataclass(frozen=True)
class SourceInfo:
    filename: str          # exact filename in data/raw_pdfs/
    madhab: str            # "Hanafi" | "Shafi'i" | "Maliki" | "Hanbali"
    scholar: str           # author of the original text
    text_title: str        # name of the work
    era: str = ""          # century / approximate date


SOURCES: list[SourceInfo] = [
    SourceInfo(
        filename="(All opinions) - Bidayat Al-Mujtahid - The Distinguished Jurists Primer - Ibn Rushd.pdf",
        madhab="Maliki",
        scholar="Ibn Rushd (Abu al-Walid Muhammad ibn Ahmad ibn Rushd)",
        text_title="Bidayat al-Mujtahid wa Nihayat al-Muqtasid (The Distinguished Jurist's Primer)",
        era="12th century",
    ),
    SourceInfo(
        filename="Ahlus Sunnah Wa Al-Jam'ah (1) A Summary of Islamic Jurisprudence - Salih Al Fawzan.pdf",
        madhab="Hanbali",
        scholar="Salih al-Fawzan",
        text_title="A Summary of Islamic Jurisprudence",
        era="contemporary (20th-21st century)",
    ),
    SourceInfo(
        filename="Ahlus Sunnah Wa Al-Jam'ah (2) Explanation of a Summary of Aqeedat Hamawiyyah - Ibn Taimiyyiah.pdf",
        madhab="Hanbali",
        scholar="Ibn Taymiyyah",
        text_title="Al-'Aqeedah al-Hamawiyyah (with explanation by Muhammad ibn Salih al-'Uthaymeen)",
        era="13th-14th century",
    ),
    SourceInfo(
        filename="Ahlus Sunnah Wa Al-Jam'ah (4) Bulugul Maram - Ibn Hajar Al-Askalani - ENG.pdf",
        madhab="Shafi'i",
        scholar="Ibn Hajar al-'Asqalani",
        text_title="Bulugh al-Maram min Adillat al-Ahkam",
        era="15th century",
    ),
    SourceInfo(
        filename="Madhab Books - Hanbali (1) Umdat al-Fiqh (2) Treatise-on-Prayer (3)  Al-Izziyyah.pdf",
        madhab="Hanbali",
        scholar="Ibn Qudamah al-Maqdisi",
        text_title="Umdat al-Fiqh",
        era="12th-13th century",
    ),
    SourceInfo(
        filename="Madhab Books - Maliki (1) The Risala (2) Fundamental Principals (3) Four Umdats (4) The Foundation.pdf",
        madhab="Maliki",
        scholar="Multiple Maliki authors",
        text_title="Collection of Maliki works including al-Risala and The Fundamental Principles of Maliki Fiqh",
        era="10th-20th century",
    ),
    SourceInfo(
        filename="Madhab Books - Shafei (1) Al-Risala (2) Minhaj (3) Umdat (4) Matn Safinat.pdf",
        madhab="Shafi'i",
        scholar="Multiple Shafi'i authors",
        text_title="Collection of Shafi'i works including al-Risala, Minhaj al-Talibin, 'Umdat al-Salik, and Safinat al-Naja",
        era="2nd-20th century",
    ),
    SourceInfo(
        filename="The-Mukhtasar-Al-Quduri-A-Manual-of-Islamic-Law-According-to-the-Hanafi-School.pdf",
        madhab="Hanafi",
        scholar="Abu al-Husayn Ahmad ibn Muhammad al-Quduri al-Baghdadi",
        text_title="Mukhtasar al-Quduri (A Manual of Islamic Law According to the Hanafi School)",
        era="10th-11th century",
    ),
    SourceInfo(
        filename="english_nur_al_idah_classical_hanafi_fiqh_manual.pdf",
        madhab="Hanafi",
        scholar="Hasan ibn 'Ammar al-Shurunbulali",
        text_title="Nur al-Idah (The Light of Clarification)",
        era="17th century",
    ),
]


VALID_MADHABS = {"Hanafi", "Shafi'i", "Maliki", "Hanbali"}


def validate_registry() -> None:
    seen_files = set()
    for s in SOURCES:
        if s.madhab not in VALID_MADHABS:
            raise ValueError(
                f"Invalid madhab '{s.madhab}' for {s.filename}. "
                f"Must be one of {VALID_MADHABS}."
            )
        if s.filename in seen_files:
            raise ValueError(f"Duplicate filename in registry: {s.filename}")
        seen_files.add(s.filename)


def get_source_by_filename(filename: str) -> SourceInfo:
    for s in SOURCES:
        if s.filename == filename:
            return s
    raise KeyError(
        f"No registry entry for '{filename}'. "
        f"Add a SourceInfo entry in source_registry.py before ingesting this file."
    )
