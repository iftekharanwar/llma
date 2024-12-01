import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Grid,
  GridItem,
  Center,
  Text,
  Spinner,
  useToast
} from '@chakra-ui/react';

const MoleculeViewer3D = ({ moleculeData, format, similarMolecules = [] }) => {
  const viewerRef = useRef(null);
  const similarViewersRef = useRef([]);
  const containerRef = useRef(null);
  const toast = useToast();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [scriptsLoaded, setScriptsLoaded] = useState(false);
  const [rdkitInitialized, setRdkitInitialized] = useState(false);
  const [currentViewer, setCurrentViewer] = useState(null);

  // Load required scripts
  useEffect(() => {
    const loadScripts = async () => {
      if (scriptsLoaded) return;

      try {
        setLoading(true);
        setError(null);

        // Load jQuery first
        if (!window.jQuery) {
          await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://code.jquery.com/jquery-3.6.0.min.js';
            script.onload = resolve;
            script.onerror = () => reject(new Error('Failed to load jQuery'));
            document.head.appendChild(script);
          });
        }

        // Load 3Dmol.js with proper initialization check
        if (!window.$3Dmol) {
          await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://3dmol.org/build/3Dmol-min.js';
            script.async = true;
            script.onload = () => {
              console.log('3Dmol.js loaded successfully');
              window.$3Dmol ? resolve() : reject(new Error('3Dmol.js failed to initialize'));
            };
            script.onerror = () => reject(new Error('Failed to load 3Dmol.js'));
            document.head.appendChild(script);
          });
        }

        // Load and initialize RDKit with proper error handling
        await new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = '/rdkit/RDKit_minimal.js';
          script.onload = () => {
            if (typeof window.initRDKitModule !== 'function') {
              reject(new Error('RDKit initialization function not found'));
              return;
            }

            // Configure RDKit module
            window.Module = {
              locateFile: (file) => `/rdkit/${file}`,
              onRuntimeInitialized: () => {
                window.initRDKitModule()
                  .then(RDKit => {
                    window.RDKit = RDKit;
                    // Verify RDKit initialization with a simple test
                    try {
                      const testMol = RDKit.get_mol('C');
                      if (testMol) {
                        testMol.delete();
                        setRdkitInitialized(true);
                        console.log('RDKit initialized successfully');
                        resolve();
                      } else {
                        throw new Error('Failed to create test molecule');
                      }
                    } catch (error) {
                      reject(new Error(`RDKit test failed: ${error.message}`));
                    }
                  })
                  .catch(error => reject(new Error(`RDKit initialization failed: ${error.message}`)));
              }
            };
          };
          script.onerror = () => reject(new Error('Failed to load local RDKit script'));
          document.head.appendChild(script);
        });

        setScriptsLoaded(true);
        console.log('All scripts loaded successfully');
      } catch (error) {
        console.error('Error loading scripts:', error);
        setError(`Failed to initialize visualization: ${error.message}`);
        toast({
          title: 'Initialization Error',
          description: error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setLoading(false);
      }
    };

    loadScripts();
  }, [scriptsLoaded, toast]);

  const convertSMILEStoMOL = async (smiles) => {
    if (!window.RDKit) {
      throw new Error('RDKit not initialized');
    }

    try {
      const mol = window.RDKit.get_mol(smiles);
      if (!mol) {
        throw new Error('Failed to create molecule from SMILES');
      }

      try {
        // Add hydrogens before 3D coordinate generation
        mol.add_hs_in_place();

        // Generate multiple conformers using ETKDG
        const params = {
          randomSeed: 42,
          maxIterations: 2000,
          useRandomCoords: true,
          pruneRmsThresh: 0.1,
          numThreads: 4,
          enforceChirality: true
        };

        const embedResult = mol.EmbedMultipleConfs(params);
        if (embedResult === -1) {
          throw new Error('Failed to generate conformers');
        }

        // Optimize all conformers using MMFF94s force field
        const props = new window.RDKit.VectorUns();
        for (let i = 0; i < mol.numConformers(); i++) {
          props.push_back(i);
        }
        mol.MMFFOptimizeMoleculeConfs(props);
        props.delete();

        // Get the lowest energy conformer
        const molBlock = mol.get_molblock();
        if (!molBlock) {
          throw new Error('Failed to generate MOL block');
        }

        return molBlock;
      } finally {
        mol.delete();
      }
    } catch (error) {
      console.error('Error in SMILES conversion:', error);
      throw new Error(`Failed to convert SMILES: ${error.message}`);
    }
  };

  const initViewer = async (format, data) => {
    try {
      // Clear any existing viewer
      if (viewerRef.current) {
        viewerRef.current.innerHTML = '';
      }

      // Create new viewer container
      const container = document.getElementById('molecule-viewer-container');
      if (!container) {
        throw new Error('Viewer container not found');
      }

      console.log('Initializing viewer with format:', format, 'container:', container);

      // Initialize 3Dmol viewer
      const config = {
        backgroundColor: '#0D121E',
        antialias: true,
        id: 'molecule-viewer-container'
      };

      const viewer = window.$3Dmol.createViewer(container, config);
      if (!viewer) {
        throw new Error('Failed to create 3Dmol viewer');
      }

      // Set viewer properties
      viewer.setBackgroundColor('#0D121E');
      viewer.resize();

      // Handle different molecule formats
      switch (format) {
        case 'SMILES':
          const mol = await convertSMILEStoMOL(data);
          viewer.addModel(mol, "mol");
          viewer.setStyle({}, {
            stick: { radius: 0.15 },
            sphere: { scale: 0.25 }
          });
          break;

        case 'MOL':
          viewer.addModel(data, "mol");
          viewer.setStyle({}, {
            stick: { radius: 0.15 },
            sphere: { scale: 0.25 }
          });
          break;

        case 'PDB':
          viewer.addModel(data, "pdb");
          viewer.setStyle({}, {
            cartoon: {
              color: 'spectrum',
              thickness: 1.0,
              opacity: 0.95
            },
            stick: {
              radius: 0.15,
              colorscheme: 'rasmol'
            }
          });
          break;

        default:
          throw new Error(`Unsupported format: ${format}`);
      }

      // Center and zoom
      viewer.zoomTo();
      viewer.render();

      console.log('Viewer initialized successfully');
      return viewer;
    } catch (error) {
      console.error('Error in initViewer:', error);
      throw error;
    }
  };

  // Initialize viewers
  useEffect(() => {
    const initializeViewers = async () => {
      if (!scriptsLoaded || !rdkitInitialized) {
        return; // Wait for scripts to load
      }

      try {
        if (!moleculeData) {
          return;
        }

        setLoading(true);
        setError(null);

        // Create viewer container if it doesn't exist
        if (!viewerRef.current) {
          const container = document.createElement('div');
          container.id = 'molecule-viewer-container';
          container.style.width = '100%';
          container.style.height = '100%';
          container.style.minHeight = '400px'; // Ensure minimum height
          containerRef.current.appendChild(container);
          viewerRef.current = container;

          // Wait for container to be properly sized
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Validate input format
        const supportedFormats = ['SMILES', 'MOL', 'PDB', 'SDF'];
        const normalizedFormat = format?.toUpperCase();
        if (!supportedFormats.includes(normalizedFormat)) {
          throw new Error(`Unsupported molecule format: ${format}`);
        }

        try {
          // Initialize main viewer
          await initViewer(normalizedFormat, moleculeData);

          // Initialize similar molecule viewers if available
          if (similarMolecules?.length > 0) {
            similarViewersRef.current = [];
            await Promise.all(similarMolecules.map(async (mol, index) => {
              const container = document.getElementById(`similar-mol-${index}`);
              if (container && mol.structure) {
                try {
                  const similarViewer = await initViewer(
                    mol.format?.toUpperCase() || normalizedFormat,
                    mol.structure
                  );
                  similarViewersRef.current[index] = similarViewer;
                } catch (error) {
                  console.error(`Failed to initialize similar viewer ${index}:`, error);
                  toast({
                    title: 'Warning',
                    description: `Failed to load similar molecule ${index + 1}: ${error.message}`,
                    status: 'warning',
                    duration: 5000,
                    isClosable: true,
                  });
                }
              }
            }));
          }
        } catch (error) {
          throw new Error(`Visualization failed: ${error.message}`);
        }
      } catch (error) {
        console.error('Error in viewer initialization:', error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    initializeViewers();
  }, [moleculeData, similarMolecules, format, scriptsLoaded, rdkitInitialized, toast]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (viewerRef.current) {
        viewerRef.current.clear();
      }
      similarViewersRef.current.forEach(viewer => {
        if (viewer) {
          viewer.clear();
        }
      });
    };
  }, []);

  return (
    <Box>
      <Grid templateColumns={{ base: "1fr", lg: "3fr 2fr" }} gap={8}>
        {/* Main molecule viewer */}
        <GridItem>
          <Box
            position="relative"
            width="100%"
            height="600px"
            borderRadius="2xl"
            overflow="hidden"
            bg="#0D121E"
            border="1px solid"
            borderColor="whiteAlpha.200"
            _hover={{ borderColor: 'blue.400' }}
            transition="all 0.3s"
            boxShadow="0 8px 16px rgba(0, 0, 0, 0.2)"
          >
            {loading && (
              <Center
                position="absolute"
                top="0"
                left="0"
                right="0"
                bottom="0"
                backgroundColor="rgba(13, 18, 30, 0.95)"
                zIndex="1"
                borderRadius="2xl"
              >
                <Box textAlign="center">
                  <Spinner size="xl" color="blue.400" thickness="4px" speed="0.8s" />
                  <Text color="blue.200" mt={4} fontSize="md" fontWeight="medium">
                    Initializing 3D visualization...
                  </Text>
                </Box>
              </Center>
            )}
            {error && (
              <Center
                position="absolute"
                top="0"
                left="0"
                right="0"
                bottom="0"
                backgroundColor="rgba(13, 18, 30, 0.95)"
                zIndex="1"
                borderRadius="2xl"
                p={8}
              >
                <Box textAlign="center" maxW="md">
                  <Text color="red.400" fontSize="lg" fontWeight="semibold" mb={3}>
                    {error}
                  </Text>
                  <Text color="gray.300" fontSize="md">
                    Please ensure molecule data is in the correct format and try again.
                  </Text>
                </Box>
              </Center>
            )}
            <Box
              ref={containerRef}
              id="molecule-viewer-container"
              width="100%"
              height="100%"
              backgroundColor="#0D121E"
              borderRadius="2xl"
              overflow="hidden"
              position="relative"
              data-testid="molecule-viewer-3d"
              sx={{
                '& canvas': {
                  borderRadius: '2xl',
                  outline: 'none'
                }
              }}
            />
          </Box>
        </GridItem>

        {/* Similar molecules grid */}
        <GridItem>
          <Grid templateColumns="1fr" gap={6}>
            {similarMolecules?.map((mol, index) => (
              <Box
                key={index}
                id={`similar-mol-${index}`}
                width="100%"
                height="180px"
                backgroundColor="#0D121E"
                border="1px solid"
                borderColor="whiteAlpha.200"
                borderRadius="xl"
                overflow="hidden"
                position="relative"
                _hover={{
                  borderColor: 'blue.400',
                  transform: 'translateY(-2px)',
                  boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)'
                }}
                transition="all 0.3s"
              >
                <Text
                  position="absolute"
                  top={3}
                  left={4}
                  color="blue.200"
                  fontSize="sm"
                  fontWeight="semibold"
                  bg="rgba(13, 18, 30, 0.9)"
                  px={3}
                  py={1}
                  borderRadius="md"
                  letterSpacing="wide"
                >
                  Similar Structure {index + 1}
                </Text>
              </Box>
            ))}
          </Grid>
        </GridItem>
      </Grid>
    </Box>
  );
};

export default MoleculeViewer3D;
