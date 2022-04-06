# BSD 3-Clause License

# Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc.
# and PyData Development Team
# All rights reserved.

# Copyright (c) 2011-2021, Open source contributors.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Function copied verbatim from https://github.com/pandas-dev/pandas/blob/
# master/pandas/_libs/tslibs/parsing.pyx with some small adaption.
#
# We use this function to retrieve the format strings from dates.
# However, as it is a private function in the pandas library,
# relying on backwards compatibility is not practical.
# Thus, we port the class over so that both issues are resolved.

import re

from dateutil.parser import parse as du_parse


# Class copied verbatim from https://github.com/dateutil/dateutil/pull/732
#
# We use this class to parse and tokenize date strings. However, as it is
# a private class in the dateutil library, relying on backwards compatibility
# is not practical. In fact, using this class issues warnings (xref gh-21322).
# Thus, we port the class over so that both issues are resolved.
#
# Copyright (c) 2017 - dateutil contributors
class _timelex:
    def __init__(self, instream):
        if getattr(instream, 'decode', None) is not None:
            instream = instream.decode()

        if isinstance(instream, str):
            self.stream = instream
        elif getattr(instream, 'read', None) is None:
            raise TypeError(
                'Parser must be a string or character stream, not '
                f'{type(instream).__name__}')
        else:
            self.stream = instream.read()

    def get_tokens(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.
        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it i"2020/01/15 -- 12:00:21")s necessary to
        read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """

        stream = self.stream.replace('\x00', '')

        # TODO: Change \s --> \s+ (this doesn't match existing behavior)
        # TODO: change the punctuation block to punc+ (does not match existing)
        # TODO: can we merge the two digit patterns?
        tokens = re.findall(r"\s|"
                            r"(?<![\.\d])\d+\.\d+(?![\.\d])"
                            r"|\d+"
                            r"|[a-zA-Z]+"
                            r"|[\./:]+"
                            r"|[^\da-zA-Z\./:\s]+", stream)

        # Re-combine token tuples of the form ["59", ",", "456"] because
        # in this context the "," is treated as a decimal
        # (e.g. in python's default logging format)
        for n, token in enumerate(tokens[:-2]):
            # Kludge to match ,-decimal behavior; it'd be better to do this
            # later in the process and have a simpler tokenization
            if (token is not None and token.isdigit() and
                    tokens[n + 1] == ',' and tokens[n + 2].isdigit()):
                # Have to check None b/c it might be replaced during the loop
                # TODO: I _really_ don't faking the value here
                tokens[n] = token + '.' + tokens[n + 2]
                tokens[n + 1] = None        # cdef:
        #     Py_ssize_t n
        tokens = [x for x in tokens if x is not None]
        return tokens

    @classmethod
    def split(cls, s):
        return cls(s).get_tokens()


_DATEUTIL_LEXER_SPLIT = _timelex.split


def guess_datetime_format(
    dt_str,
    dayfirst=False,
    dt_str_parse=du_parse,
    dt_str_split=_DATEUTIL_LEXER_SPLIT,
):
    """
    Guess the datetime format of a given datetime string.
    Parameters
    ----------
    dt_str : str
        Datetime string to guess the format of.
    dayfirst : bool, default False
        If True parses dates with the day first, eg 20/01/2005
        Warning: dayfirst=True is not strict, but will prefer to parse
        with day first (this is a known bug).
    dt_str_parse : function, defaults to `dateutil.parser.parse`
        This function should take in a datetime string and return
        a `datetime.datetime` guess that the datetime string represents
    dt_str_split : function, defaults to `_DATEUTIL_LEXER_SPLIT` (dateutil)
        This function should take in a datetime string and return
        a list of strings, the guess of the various specific parts
        e.g. '2011/12/30' -> ['2011', '/', '12', '/', '30']
    Returns
    -------
    ret : datetime format string (for `strftime` or `strptime`)
    """
    if dt_str_parse is None or dt_str_split is None:
        return None

    if not isinstance(dt_str, str):
        return None

    day_attribute_and_format = (('day',), '%d', 2)

    # attr name, format, padding (if any)
    datetime_attrs_to_format = [
        (('year', 'month', 'day'), '%Y%m%d', 0),
        (('year',), '%Y', 0),
        (('month',), '%B', 0),
        (('month',), '%b', 0),
        (('month',), '%m', 2),
        day_attribute_and_format,
        (('hour',), '%H', 2),
        (('minute',), '%M', 2),
        (('second',), '%S', 2),
        (('microsecond',), '%f', 6),
        (('second', 'microsecond'), '%S.%f', 0),
        (('tzinfo',), '%z', 0),
        (('tzinfo',), '%Z', 0),
        (('day_of_week',), '%a', 0),
        (('day_of_week',), '%A', 0),
        (('meridiem',), '%p', 0),
    ]

    if dayfirst:
        datetime_attrs_to_format.remove(day_attribute_and_format)
        datetime_attrs_to_format.insert(0, day_attribute_and_format)

    try:
        parsed_datetime = dt_str_parse(dt_str, dayfirst=dayfirst)
    except (ValueError, OverflowError):
        # In case the datetime can't be parsed, its format cannot be guessed
        return None

    if parsed_datetime is None:
        return None

    # the default dt_str_split from dateutil will never raise here; we assume
    #  that any user-provided function will not either.
    tokens = dt_str_split(dt_str)

    # Normalize offset part of tokens.
    # There are multiple formats for the timezone offset.
    # To pass the comparison condition between the output of `strftime` and
    # joined tokens, which is carried out at the final step of the function,
    # the offset part of the tokens must match the '%z' format like '+0900'
    # instead of ‘+09:00’.
    if parsed_datetime.tzinfo is not None:
        offset_index = None
        if len(tokens) > 0 and tokens[-1] == 'Z':
            # the last 'Z' means zero offset
            offset_index = -1
        elif len(tokens) > 1 and tokens[-2] in ('+', '-'):
            # ex. [..., '+', '0900']
            offset_index = -2
        elif len(tokens) > 3 and tokens[-4] in ('+', '-'):
            # ex. [..., '+', '09', ':', '00']
            offset_index = -4

        if offset_index is not None:
            # If the input string has a timezone offset like '+0900',
            # the offset is separated into two tokens, ex. ['+', '0900’].
            # This separation will prevent subsequent processing
            # from correctly parsing the time zone format.
            # So in addition to the format nomalization, we rejoin them here.
            tokens[offset_index] = parsed_datetime.strftime("%z")
            tokens = tokens[:offset_index + 1 or None]

    format_guess = [None] * len(tokens)
    found_attrs = set()

    for attrs, attr_format, padding in datetime_attrs_to_format:
        # If a given attribute has been placed in the format string, skip
        # over other formats for that same underlying attribute (IE, month
        # can be represented in multiple different ways)
        if set(attrs) & found_attrs:
            continue

        if parsed_datetime.tzinfo is None and attr_format in ("%Z", "%z"):
            continue

        parsed_formatted = parsed_datetime.strftime(attr_format)
        for i, token_format in enumerate(format_guess):
            token_filled = tokens[i].zfill(padding)
            if attrs == ('second', 'microsecond') and len(token_filled) > 9:
                # Ignore nanoseconds in comparison.
                is_nano_seconds = token_filled[:9] == parsed_formatted
            else:
                is_nano_seconds = False

            if token_format is None and (token_filled == parsed_formatted
                                         or is_nano_seconds):
                format_guess[i] = attr_format
                tokens[i] = token_filled
                found_attrs.update(attrs)
                break

    # Only consider it a valid guess if we have a year, month and day
    if len({'year', 'month', 'day'} & found_attrs) != 3:
        return None

    output_format = []
    for i, guess in enumerate(format_guess):
        if guess is not None:
            # Either fill in the format placeholder (like %Y)
            output_format.append(guess)
        else:
            # Or just the token separate (IE, the dashes in "01-01-2013")
            try:
                # If the token is numeric, then we likely didn't parse it
                # properly, so our guess is wrong
                float(tokens[i])
                return None
            except ValueError:
                pass

            output_format.append(tokens[i])

    guessed_format = ''.join(output_format)

    # rebuild string, capturing any inferred padding
    dt_str = ''.join(tokens)
    if parsed_datetime.strftime(guessed_format) == dt_str:
        return guessed_format
    # Check the case for nanoseconds.
    elif parsed_datetime.strftime(guessed_format) == dt_str[:-3]:
        return guessed_format
    else:
        return None
