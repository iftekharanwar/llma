import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  colors: {
    navy: {
      50: '#E6E8F4',
      100: '#BEC3E3',
      200: '#959ED1',
      300: '#6B79BE',
      400: '#4C5EAF',
      500: '#2D43A0',
      600: '#243480',
      700: '#1B2660',
      800: '#121940',
      900: '#090C20',
    },
  },
  styles: {
    global: {
      body: {
        bg: 'rgb(13, 18, 30)',
        color: 'gray.100',
      },
    },
  },
  components: {
    Button: {
      variants: {
        solid: {
          bg: 'blue.500',
          color: 'white',
          _hover: {
            bg: 'blue.600',
          },
        },
        outline: {
          borderColor: 'whiteAlpha.300',
          color: 'white',
          _hover: {
            bg: 'whiteAlpha.100',
          },
        },
      },
    },
    Input: {
      variants: {
        filled: {
          field: {
            bg: 'whiteAlpha.50',
            borderColor: 'whiteAlpha.200',
            _hover: {
              bg: 'whiteAlpha.100',
            },
            _focus: {
              bg: 'whiteAlpha.100',
              borderColor: 'blue.400',
            },
          },
        },
      },
      defaultProps: {
        variant: 'filled',
      },
    },
    Textarea: {
      variants: {
        filled: {
          bg: 'whiteAlpha.50',
          borderColor: 'whiteAlpha.200',
          _hover: {
            bg: 'whiteAlpha.100',
          },
          _focus: {
            bg: 'whiteAlpha.100',
            borderColor: 'blue.400',
          },
        },
      },
      defaultProps: {
        variant: 'filled',
      },
    },
    Progress: {
      baseStyle: {
        track: {
          bg: 'whiteAlpha.200',
        },
      },
    },
    Table: {
      variants: {
        simple: {
          th: {
            color: 'gray.300',
            borderColor: 'whiteAlpha.300',
            padding: '1rem',
            fontSize: 'sm',
            fontWeight: 'semibold',
          },
          td: {
            borderColor: 'whiteAlpha.200',
            padding: '1rem',
            fontSize: 'sm',
          },
          caption: {
            color: 'gray.400',
            fontSize: 'sm',
          },
        },
      },
    },
    Card: {
      baseStyle: {
        container: {
          bg: 'rgba(13, 18, 30, 0.8)',
          backdropFilter: 'blur(10px)',
          borderRadius: 'xl',
          padding: '6',
          border: '1px solid',
          borderColor: 'whiteAlpha.200',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          _hover: {
            borderColor: 'blue.400',
            transform: 'translateY(-2px)',
            transition: 'all 0.2s ease-in-out'
          },
        },
      },
    },
    Stat: {
      baseStyle: {
        container: {
          padding: '4',
        },
        label: {
          color: 'gray.400',
          fontSize: 'sm',
          fontWeight: 'medium',
        },
        number: {
          color: 'white',
          fontSize: 'xl',
          fontWeight: 'bold',
        },
        helpText: {
          color: 'gray.400',
          fontSize: 'xs',
        },
      },
    },
  },
});

export default theme;
