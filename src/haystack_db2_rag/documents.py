"""Sample banking-product corpus.

Mirrors the tutorial's domain: financial product information with metadata that
compliance filters can key off (product_type, region, risk_level, min_balance).
"""

from haystack import Document

SAMPLE_DOCUMENTS: list[Document] = [
    Document(
        content=(
            "The Everyday Savings Account pays 2.1% APY with no minimum balance and no monthly "
            "maintenance fee. Interest is compounded daily and credited monthly. Withdrawals are "
            "limited to six per statement cycle."
        ),
        meta={"product_type": "savings", "region": "US", "risk_level": "low", "min_balance": 0},
    ),
    Document(
        content=(
            "The Premier High-Yield Savings Account pays 4.35% APY but requires a $25,000 minimum "
            "daily balance. Falling below the minimum triggers a $15 monthly fee and the rate "
            "drops to the Everyday Savings tier."
        ),
        meta={"product_type": "savings", "region": "US", "risk_level": "low", "min_balance": 25000},
    ),
    Document(
        content=(
            "The Student Checking Account has no minimum balance, no overdraft fee on the first "
            "two occurrences per year, and reimburses up to $10 per month in out-of-network ATM "
            "fees. Eligibility requires proof of enrollment and an age between 17 and 24."
        ),
        meta={"product_type": "checking", "region": "US", "risk_level": "low", "min_balance": 0},
    ),
    Document(
        content=(
            "The Business Operating Account includes 200 free transactions per month, integrated "
            "payroll, and same-day ACH origination. Transactions beyond the included allowance "
            "cost $0.40 each. A $5,000 average balance waives the $30 monthly fee."
        ),
        meta={"product_type": "checking", "region": "US", "risk_level": "low", "min_balance": 5000},
    ),
    Document(
        content=(
            "The 12-Month Fixed Term Deposit pays 4.9% at maturity. Early withdrawal forfeits 90 "
            "days of interest. Deposits are insured up to the statutory limit and the principal "
            "is never at risk."
        ),
        meta={"product_type": "deposit", "region": "EU", "risk_level": "low", "min_balance": 1000},
    ),
    Document(
        content=(
            "The Balanced Growth Portfolio allocates 60% equities and 40% investment-grade bonds. "
            "Historical volatility is moderate and the product is not principal-protected. "
            "Suitability assessment is mandatory before subscription under MiFID II."
        ),
        meta={"product_type": "investment", "region": "EU", "risk_level": "medium", "min_balance": 10000},
    ),
    Document(
        content=(
            "The Emerging Markets Equity Fund targets long-term capital appreciation across Asian "
            "and Latin American markets. Currency risk and political risk are material. The fund "
            "may lose more than 30% of its value in a single year."
        ),
        meta={"product_type": "investment", "region": "APAC", "risk_level": "high", "min_balance": 5000},
    ),
    Document(
        content=(
            "The 30-Year Fixed Rate Mortgage is offered at 6.4% APR with 20% down. Private "
            "mortgage insurance is required below 20% equity. Rate locks are valid for 60 days "
            "and are extendable once for a 0.125% fee."
        ),
        meta={"product_type": "mortgage", "region": "US", "risk_level": "medium", "min_balance": 0},
    ),
    Document(
        content=(
            "The Small Business Term Loan provides $25,000 to $500,000 over two to seven years at "
            "rates from 8.5% APR. Underwriting requires two years of filed accounts and a debt "
            "service coverage ratio above 1.25."
        ),
        meta={"product_type": "loan", "region": "US", "risk_level": "medium", "min_balance": 0},
    ),
    Document(
        content=(
            "The Travel Rewards Credit Card earns 3x points on airfare and hotels and 1x "
            "elsewhere, with no foreign transaction fee. The annual fee is $95, waived the first "
            "year. Cash advances accrue interest immediately at 27.9% APR."
        ),
        meta={"product_type": "credit_card", "region": "US", "risk_level": "medium", "min_balance": 0},
    ),
]
