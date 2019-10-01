# Chat : This is a class that has all the logic that is used by the chatbot
# Reflections: This is a dictionary that contains a set of input values and
#               it's corresponding output values.

reflections = {
  "i am"       : "you are",
  "i was"      : "you were",
  "i"          : "you",
  "i'm"        : "you are",
  "i'd"        : "you would",
  "i've"       : "you have",
  "i'll"       : "you will",
  "my"         : "your",
  "you are"    : "I am",
  "you were"   : "I was",
  "you've"     : "I have",
  "you'll"     : "I will",
  "your"       : "my",
  "yours"      : "mine",
  "you"        : "me",
  "me"         : "you"
}

from nltk.chat.util import Chat, reflections

pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how are you today? ",]
    ],
    [
        r"what is your name ?",
        ["My name is Lucifer and I'm a chatbot ?",]
    ],
    [
        r"how are you  ?",
        ["I'm doing good\nHow bout you ?",]
    ],
    [
        r"(.*) investment options (.*)",
        ["1.Mutual Funds\n2.National Pension Scheme\n3.Public Provident Fund\n4. Real Estate Investment\nWhich one do you want to see?"]
    ],
    [
        r"(.*) mutual funds (.*)",
        ["These are considered to be one of the best avenues for investment in our country. Amongst mutual funds, equity mutual funds are in particular top-rated. Such is the earning potential of equity mutual funds that some of the best performing funds have generated Cumulative Average Growth Returns of about 20% in a decade. The point to note is that with such high rewards come high risks as well. It is advised that you consult financial experts before making any decisions. There are many types of portfolios and styles of investing, but with mutual funds, you can access the best of all and generate good income. Investment in these is straightforward, and with the you can start with as little an amount as Rs.500 a month."]
    ],
    [
        r"(.*) national pension scheme (.*)",
        ["The NPS is a government-sponsored scheme that is one of the best modes of investment for those with a very low-risk profile. As the government backs it, you don’t stand to lose your investment. Regardless of your contribution, you will receive a certain amount of pension. Apart from this, investing in the NPS qualifies you to additional tax benefits under Section 80CCD (1B). This deduction is over and above regular deductions under Section 80C, Section 80CCC and Section 80CCD where you can save up to Rs.1.5 lakh every year. With NPS, under Section 80CCD (1B), you can contribute an additional Rs.50,000. Furthermore, under Section 80CCD(2), if you are in the high tax bracket, you can have your salary structured such that your employer contributes 10% of your salary without you having to do the same."]
    ],
    [
        r"(.*) public provident fund (.*)",
        ["If you are a risk-averse investor, the Public Provident Fund (PPF) could be the one for you. PPF is one of the popular Section 80C options for the common man because it is not risky. Also, the scheme is easy to start for those who are not internet savvy. You can open this account in a bank or even in a post office. This fund is very similar to a bank Recurring Deposit (RD) but has a tenure of 15 years with the option to extend it further by five years. If you are a salaried person, you may find this an excellent way to set aside a certain sum every month to invest in PPF. If you require a loan, then you can avail one on your PPF and even make an early withdrawal after the 7th year of opening the account. One of the most attractive features of a PPF account is that the interest that you earn on this fund is free from taxation."]
    ],
    [
        r"(.*) real estate investment (.*)",
        ["Real estate is a good investment option for those who have abundant money in hand. It is an excellent option for long-term investment. The Real Estate Regulation and Development Act (RERA), which came into practice in 2016, has further boosted this market. The industry is well regulated with safety measures in place for buyers and sellers. The fast-paced development and urbanisation, the demand for real estate have witnessed a rise like ever before. The availability of accessible home loans has removed the barriers to affordability and allows buyers to save a significant amount of income tax annually until the payment of the home loan."]
    ],
]

def chatty():
    print("Welcome to LA. Lucifer will be our guide in the investments")

    chat = Chat(pairs, reflections)
    chat.converse()
 
if __name__ == '__main__':
    chatty()