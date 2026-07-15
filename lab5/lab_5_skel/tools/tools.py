# TODO: Import your tools here

try:
    from . import lucky_number_tool as lnt
except ImportError:
    import lucky_number_tool as lnt


tools = [
    # TODO: Add your tool instances here
    lnt.lucky_number_tool
]
