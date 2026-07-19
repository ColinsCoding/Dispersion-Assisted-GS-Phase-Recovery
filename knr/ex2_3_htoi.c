#include <stdio.h>
#include <ctype.h>

int htoi(const char s[])
{
    int i = 0;
    int n = 0;

    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        i = 2;

    for (; s[i] != '\0'; i++) {
        int digit;

        if (s[i] >= '0' && s[i] <= '9')
            digit = s[i] - '0';
        else if (s[i] >= 'a' && s[i] <= 'f')
            digit = s[i] - 'a' + 10;
        else if (s[i] >= 'A' && s[i] <= 'F')
            digit = s[i] - 'A' + 10;
        else
            break;

        n = 16 * n + digit;
    }

    return n;
}

int main(void)
{
    printf("%d\n", htoi("0x1A"));  // 26
    printf("%d\n", htoi("FF"));    // 255
    printf("%d\n", htoi("2b"));    // 43
}