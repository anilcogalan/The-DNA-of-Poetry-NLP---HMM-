from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

SLEEP_TIME = 2

# İlk sayfaya git
driver.get("http://siir.me/nazim-hikmet")
time.sleep(SLEEP_TIME)

# Tüm bağlantıları topla
links = [link.get_attribute('href') for link in driver.find_elements(By.XPATH, "//ul//a")]

collected_data = []

# Her bir bağlantı için döngü
for link in links:
    driver.get(link)
    time.sleep(SLEEP_TIME)

    # Şiir metnini bul ve al
    try:
        poem_text_element = driver.find_element(By.XPATH, "//pre[@class='stext']")
        poem_text = poem_text_element.text
        collected_data.append({'link': link, 'poem_text': poem_text})
    except:
        print(f"Şiir metni {link} adresinde bulunamadı.")

# WebDriver'ı kapat
driver.quit()

# Verileri JSON olarak kaydet
with open('poemsnh.json', 'w', encoding='utf-8') as f:
    json.dump(collected_data, f, ensure_ascii=False, indent=4)

print("Şiirler başarıyla kaydedildi.")