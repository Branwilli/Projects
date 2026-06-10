import React, { useState } from 'react';
import { 
    Container,
    Paper,
    Typography,
    TextField,
    Button,
    Grid,
    Link,
    Avatar,
    Box,
    InputAdornment,
    IconButton,
    FormControlLabel,
    Checkbox,
    CircularProgress,
    Snackbar,
    Alert
} from '@mui/material';
import {
    LockOutlined as LockOutlinedIcon,
    Visibility,
    VisibilityOff
} from '@mui/icons-material';

const SignupPage = () => {
    const [showPassword, setShowPassword] = useState(false);
    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        email: '',
        password: '',
        confirmPassword: '',
        acceptTerms: false
    });
    const [errors, setErrors] = useState({});
    const [loading, setLoading] = useState(false);
    const [snackbar, setSnackbar] = useState({
        open: false,
        message: '',
        severity: 'success'
    });

    const handleChange = (e) => {
        const { name, value, checked } = e.target;
        setFormData({
            ...formData, 
            [name]: name === 'acceptTerms' ? checked : value
        });
    };

    const validate = () => {
        const newErrors = {};
        if (!formData.firstName.trim()) newErrors.firstName = 'First name is required';
        if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required';
        if (!formData.email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
            newErrors.email = 'Email is invalid';
        } 
        if (!formData.password) {
            newErrors.password = 'Password is required';
        } else if (formData.password.length < 8) {
            newErrors.password = 'Password must be at least 8 characters';
        }
        if (formData.password !== formData.confirmPassword) {
            newErrors.confirmPassword = 'Passwords do not match';
        }
        if (!formData.acceptTerms) {
            newErrors.acceptTerms = 'You must accept the terms and conditions';
        }
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!validate()) return;

        setLoading(true);

        try {
            const response = await fetch('http://localhost:5000/api/send-welcome-email', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  receiver_email: formData.email,
                  first_name: formData.firstName,
                  last_name: formData.lastName
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            setSnackbar({
                open: true,
                message: data.message || 'Signup successful! Email sent.',
                severity: 'success'
            });
            
            setFormData({
                firstName: '',
                lastName: '',
                email: '',
                password: '',
                confirmPassword: '',
                acceptTerms: false
            });

        } catch (error) {
            setSnackbar({
                open: true,
                message: error.message || 'Signup failed. Please try again.',
                severity: 'error'
            });
        } finally {
            setLoading(false);
        }
    };

    const handleCloseSnackbar = () => {
        setSnackbar({ ...snackbar, open: false });
    };

    return (
        <Container component="main" maxWidth="sm">
            <Paper
                elevation={3}
                sx={{
                    mt: 8, 
                    p: 4,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}
                >
                    <Avatar sx={{ m: 1, bgcolor: 'sencondary.main' }}>
                        <LockOutlinedIcon />
                    </Avatar>
                    <Typography component='h1' variant='h5'>
                        Sign Up
                    </Typography>
                    <Box component='form' onSubmit={handleSubmit} sx={{ mt: 3 }}>
                        <Grid container spacing={2}>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    autoComplete='given-name'
                                    name="firstName"
                                    required
                                    fullWidth
                                    id='firstName'
                                    label='First Name'
                                    autoFocus
                                    value={formData.firstName}
                                    onChange={handleChange}
                                    error={!!errors.firstName}
                                    helperText={errors.firstName}
                                />
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    required
                                    fullWidth
                                    id="lastName"
                                    label="Last Name"
                                    name="lastName"
                                    autoComplete="family-name"
                                    value={formData.lastName}
                                    onChange={handleChange}
                                    error={!!errors.lastName}
                                    helperText={errors.lastName}
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    required
                                    fullWidth
                                    id="email"
                                    label="Email Address"
                                    name="email"
                                    autoComplete="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    error={!!errors.email}
                                    helperText={errors.email}
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                required
                                fullWidth
                                name='password'
                                label='Password'
                                type={showPassword ? 'text' : 'password'}
                                id='password'
                                autoComplete='new-password'
                                value={formData.password}
                                onChange={handleChange}
                                error={!!errors.password}
                                helperText={errors.password}
                                InputProps={{
                                endAdornment: (
                                    <InputAdornment position="end">
                                        <IconButton
                                            aria-label="toggle password visibility"
                                            onClick={() => setShowPassword(!showPassword)}
                                            edge="end"
                                        >
                                            {showPassword ? <VisibilityOff /> : <Visibility />}
                                        </IconButton>
                                    </InputAdornment>
                                ),
                                }}
                            />
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    required
                                    fullWidth
                                    name="confirmPassword"
                                    label="Confirm Password"
                                    type={showPassword ? 'text' : 'password'}
                                    id="confirmPassword"
                                    value={formData.confirmPassword}
                                    onChange={handleChange}
                                    error={!!errors.confirmPassword}
                                    helperText={errors.confirmPassword}
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <FormControlLabel
                                    control={
                                        <Checkbox 
                                            name="acceptTerms"
                                            color="primary" 
                                            checked={formData.acceptTerms}
                                            onChange={handleChange}
                                        />
                                    }
                                    label={
                                        <Typography>
                                          I agree to the{' '}
                                          <Link href="#" underline="hover">
                                            Terms and Conditions
                                          </Link>
                                        </Typography>
                                      }
                                    />  
                                    {errors.acceptTerms && (
                                        <Typography color="error" variant="body2">
                                            {errors.acceptTerms}
                                        </Typography>
                                    )}
                            </Grid>
                        </Grid>
                        <Button
                            type="submit"
                            fullWidth
                            variant="contained"
                            sx={{ mt: 3, mb: 2 }}
                            disabled={loading}
                            >
                            {loading ? <CircularProgress size={24} /> : 'Sign Up'}
                        </Button>
                        <Snackbar
                            open={snackbar.open}
                            autoHideDuration={6000}
                            onClose={handleCloseSnackbar}
                            >
                            <Alert
                                onClose={handleCloseSnackbar}
                                severity={snackbar.severity}
                                sx={{ width: '100%' }}
                            >
                                {snackbar.message}
                            </Alert>
                        </Snackbar>
                    </Box>
                </Paper>
        </Container>
    );
};

export default SignupPage;