#!/usr/bin/env python3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import time
#some standard imports

#getOptions() function is used to set parameters for our CromeDriver
def getOptions():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--start-fullscreen')
    # options.add_argument('--headless')
    
    #____________________________________
    #The following options are here, so that the usual websites don#t detect us as a headless Browser, since many will try to block it
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
    options.add_argument('user-agent={0}'.format(user_agent))
    
    #____________________________________
    options.add_argument('--no-sandbox')
    
    #These following options vary from user to user dependant on the Errors, which the driver causes due to GUI
    options.add_argument('--log-level=1')
    options.add_argument("--disable-3d-apis")
    return options

#startSession() is used to keep the browser open through global methods 
#without using this the headless chrome browser will shortly open and close again, making ti impossible to scrape

def startSession():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=getOptions())
    #initializes the user agent, so we don't get blocked
    driver.execute_script("return navigator.userAgent")
    return driver
  
if __name__ == "__main__":
    driver = None
    try:
        driver: webdriver.Chrome = startSession()
        driver.get('https://uhhgpt.uni-hamburg.de/interface.php')
        
        # Get Credentials
        import os
        from dotenv import load_dotenv
        load_dotenv()
        username = os.getenv('UHH_USERNAME')
        password = os.getenv('UHH_PASSWORD')
        print(username, password)

        # Click on the Login Button
        try:
            login_button = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, '/html/body/div/aside/div[1]/form/button'))
            login_button.click()
        except TimeoutException:
            print("login_button couldn't be found")

        # Fill Out Credentials
        try:
            username_field = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, '//*[@id="username"]'))
            password_field = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, '//*[@id="password"]'))
            username_field.send_keys(username)
            password_field.send_keys(password)
        except TimeoutException:
            print("Inputs couldn't be found")
        
        # Log in
        try:
            confirm_button = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, '//*[@id="inhalt"]/main/section[1]/article/div[2]/div/div/div[1]/div[1]/form/div[2]/div/button'))
            confirm_button.click()
            data_prot_button = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, '//*[@id="data-protection"]/div/button'))
            data_prot_button.click()
        except TimeoutException:
            print("Buttons couldn't be found")

        # Init
        try:
            init_text = '''
            Du bist ein professioneller Autor, welcher Wetterberichte und Flachwitze schreibt. 
            Nun möchtest du beides kombinieren und deine Wetterberichte in lustige, und durchaus etwas unseriös klingende Wetterberichte verwandeln. 
            Schreibe bitte folgende Wetterbericht um, die ich dir geben werde'''
            chat_input = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, '//*[@id="texreachat"]'))
            chat_input.send_keys()
            send_button = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.XPATH, '/html/body/div/div[2]/div[3]/div[1]/div[2]'))
            send_button.click()
        except TimeoutException:
            print("Chat couldn't be found")

        while True:
            # Perform tasks
            pass  # Replace with your code
    except KeyboardInterrupt:
        pass
    finally:
        if driver is not None:
            driver.quit()